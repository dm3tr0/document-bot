import asyncio
import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import APIError, RateLimitError

from ..config import settings
from . import db, llm
from .db import WHOLE_FOREST
from .extract import ALLOWED_EXTENSIONS, OCRUnavailableError, extract_document
from .knowledge import tokenize, words_match

log = logging.getLogger(__name__)

UPDATE_CANDIDATES = 8
COMPARISON_BATCH_SIZE = 4

# What the AI (or a user) may answer when the forest part is not clear.
UNKNOWN_WORDS = {"unknown", "невідомо", "не визначено", "не зрозуміло", "без категорії", "none", "n/a", "-"}


class IngestError(Exception):
    """Carries a user-facing message (in Ukrainian)."""


@dataclass
class Sender:
    id: int | None
    name: str
    username: str | None = None
    uploaded_at: str | None = None


@dataclass
class Result:
    doc_id: int
    title: str
    filename: str
    category: str
    summary: str
    duplicate: bool = False
    updated: bool = False


def clean_title(raw: str, limit: int = 80) -> str:
    """Make a string safe to use as a file or folder name (also on Windows)."""
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", raw)
    title = re.sub(r"\s+", " ", title).strip(" .")
    return title[:limit].strip(" .")


def normalize_category(raw: str, existing: list[str]) -> str | None:
    """Tidy a category name. None means 'unclear'. Reuses an existing name that differs only in case."""
    name = clean_title(raw, limit=40)
    if not name or name.casefold() in UNKNOWN_WORDS:
        return None
    if name.casefold() == WHOLE_FOREST.casefold():
        return WHOLE_FOREST
    name = name[0].upper() + name[1:]
    return next((c for c in existing if c.casefold() == name.casefold()), name)


def _free_path(folder: Path, title: str, ext: str, n: int) -> Path:
    return folder / (f"{title}{ext}" if n == 1 else f"{title} ({n}){ext}")


def save_file(folder: Path, title: str, ext: str, data: bytes) -> Path:
    """Write data to folder/<title><ext>, adding ' (2)', ' (3)'... if the name is taken."""
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = _free_path(folder, title, ext, n)
        try:
            with open(path, "xb") as f:  # "x" = fail if exists, so we never overwrite
                f.write(data)
            return path
        except FileExistsError:
            n += 1


def _duplicate(row: sqlite3.Row) -> Result:
    return Result(row["id"], row["title"], row["filename"], row["category"], row["summary"], duplicate=True)


async def find_update(text: str, filename: str) -> sqlite3.Row | None:
    """Shortlist by word overlap, then let AI check purpose and exact location."""
    documents = await asyncio.to_thread(db.list_documents)
    query_words = set(tokenize(filename + " " + text))

    def similarity(row):
        title_words = set(tokenize(row["title"]))
        body_words = set(tokenize(row["text"]))
        return sum(
            (3 if word in title_words else 1)
            for word in title_words | body_words
            if any(words_match(word, query) for query in query_words)
        )
    # Shortlist related advice before AI comparison to avoid sending the whole
    # archive on every upload. Exact scope is checked by the model, not category.
    ranked = sorted(((similarity(row), row) for row in documents),
                    key=lambda item: item[0], reverse=True)
    related = [row for score, row in ranked[:UPDATE_CANDIDATES] if score > 0]
    for offset in range(0, len(related), COMPARISON_BATCH_SIZE):
        candidates = related[offset:offset + COMPARISON_BATCH_SIZE]
        doc_id = await llm.find_updated_document(text, filename, candidates)
        if doc_id is not None:
            return next(row for row in candidates if row["id"] == doc_id)
    return None


async def ingest(data: bytes, original_name: str, sender: Sender) -> Result:
    """Extract, classify and compare advice, then insert or replace a document.

    Exact duplicates skip AI. Changed content needs a semantic match before an
    existing record can be replaced; an unclear category defaults to WHOLE_FOREST.
    """
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError("Підтримуються лише файли .txt, .pdf та .docx.")

    content_hash = hashlib.sha256(data).hexdigest()
    existing_doc = await asyncio.to_thread(db.find_by_hash, content_hash)
    if existing_doc:
        await asyncio.to_thread(db.refresh_uploaded_at, existing_doc["id"], sender.uploaded_at)
        return _duplicate(existing_doc)

    try:
        text = await extract_document(data, ext)
    except OCRUnavailableError as exc:
        raise IngestError("Не налаштовано розпізнавання сканів. Зверніться до адміністратора.") from exc
    except (APIError, llm.TranscriptionError) as exc:
        log.warning("Document vision extraction failed: %s", type(exc).__name__)
        raise IngestError("Не вдалося розпізнати скан. Спробуйте пізніше.") from exc
    except Exception as exc:  # corrupt, encrypted, not really a pdf/docx...
        raise IngestError("Не вдалося прочитати файл. Можливо, він пошкоджений або захищений паролем.") from exc
    if not text.strip():
        raise IngestError("У файлі немає тексту, який я можу прочитати.")

    categories = await asyncio.to_thread(db.list_categories)
    try:
        raw_category, ai_title, summary = await llm.classify(text, categories, filename=original_name)
    except (APIError, llm.ClassificationError):
        log.warning("Document analysis failed; saved in whole-forest category")
        raw_category, ai_title, summary = "", "", ""
    try:
        previous = await find_update(text, original_name)
    except (APIError, llm.ClassificationError) as exc:
        raise IngestError("Не вдалося перевірити, чи документ оновлює наявні правила. Спробуйте пізніше.") from exc
    category = normalize_category(raw_category, categories) or WHOLE_FOREST

    first_line = next((line for line in text.splitlines() if line.strip()), "")
    stem = clean_title(Path(original_name).stem)
    title = clean_title(ai_title) or stem or clean_title(first_line) or "Документ"

    # Save the new file first; remove the previous file only after the DB commits.
    path = await asyncio.to_thread(save_file, settings.storage_dir, title, ext, data)
    try:
        fields = dict(
            title=title, filename=path.name, category=category, summary=summary, text=text,
            content_hash=content_hash, sender_id=sender.id, sender_name=sender.name,
            sender_username=sender.username, from_archive=False, uploaded_at=sender.uploaded_at,
        )
        if previous is None:
            doc_id = await asyncio.to_thread(db.add_document, **fields)
        else:
            doc_id = previous["id"]
            changed = await asyncio.to_thread(
                db.update_document, doc_id, expected_hash=previous["content_hash"], **fields
            )
            if not changed:
                raise IngestError("У базі вже є новіше оновлення цього документа. Надішліть файл повторно, якщо потрібно.")
    except sqlite3.IntegrityError:  # the very same file was added a moment ago
        path.unlink(missing_ok=True)
        existing_doc = await asyncio.to_thread(db.find_by_hash, content_hash)
        if existing_doc is None:
            raise
        await asyncio.to_thread(db.refresh_uploaded_at, existing_doc["id"], sender.uploaded_at)
        return _duplicate(existing_doc)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if previous is not None and not previous["from_archive"]:
        old_path = settings.storage_dir / previous["filename"]
        try:
            await asyncio.to_thread(old_path.unlink, missing_ok=True)
        except OSError:
            log.warning("Old document file could not be removed after update")
    return Result(doc_id, title, path.name, category, summary,
                  updated=previous is not None)


async def import_archive() -> None:
    """Index the files of the archive folder: the AI picks a category for each, the database stores it.

    The files themselves stay in the archive folder: nothing is copied or moved. New files are added,
    files that disappeared from the folder are removed. A file the AI could not analyse is skipped and
    retried on the next start; general documents use WHOLE_FOREST.
    Safe to run on every start.
    """
    try:
        folder = settings.archive_dir
        if not folder.is_dir():
            log.info("No archive folder at %s, skipping import", folder)
            return
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
        present: set[str] = set()
        added = 0
        for path in files:
            data = await asyncio.to_thread(path.read_bytes)
            content_hash = hashlib.sha256(data).hexdigest()
            present.add(content_hash)
            existing_doc = await asyncio.to_thread(db.find_by_hash, content_hash)
            if existing_doc:
                continue
            try:
                text = await extract_document(data, path.suffix.lower())
            except RateLimitError:
                log.warning("Archive import paused after vision rate limit; retry on next start")
                return
            except (APIError, llm.TranscriptionError) as exc:
                log.warning("Archive file %s skipped, vision extraction failed: %s", path.name, type(exc).__name__)
                continue
            except OCRUnavailableError as exc:
                log.warning("Archive file %s skipped: %s", path.name, exc)
                continue
            except Exception:
                log.warning("Archive file %s could not be read (check PDF password or file integrity)", path.name)
                continue
            if not text.strip():
                log.warning("Archive file %s has no readable text, skipped", path.name)
                continue

            categories = await asyncio.to_thread(db.list_categories)  # includes ones created a moment ago
            try:
                await asyncio.sleep(1)
                previous = await find_update(text, path.name)
                if previous is not None:
                    # An old archive file must not reintroduce replaced rules at restart.
                    if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) <= datetime.fromisoformat(previous["uploaded_at"]):
                        if previous["from_archive"]:
                            present.add(previous["content_hash"])
                        continue
                raw_category, _, summary = await llm.classify(text, categories, filename=path.name)
            except RateLimitError:
                # Stop this batch rather than exhausting retries again for every file.
                # Leave existing rows untouched; pending files retry at next startup.
                log.warning("Archive import paused after rate limit; remaining files retry on next start")
                return
            except llm.ClassificationError as exc:
                log.warning("Archive file %s skipped, invalid classification (retry on next start): %s", path.name, exc)
                continue
            except APIError as exc:
                log.warning("Archive file %s skipped, AI unavailable (retry on next start): %s", path.name, type(exc).__name__)
                continue
            category = normalize_category(raw_category, categories) or WHOLE_FOREST

            try:
                fields = dict(
                    title=path.stem, filename=path.name, category=category, summary=summary, text=text,
                    content_hash=content_hash, sender_id=None, sender_name=db.ARCHIVE,
                    sender_username=None, from_archive=True,
                    uploaded_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                )
                if previous is None:
                    await asyncio.to_thread(db.add_document, **fields)
                else:
                    changed = await asyncio.to_thread(
                        db.update_document, previous["id"], expected_hash=previous["content_hash"], **fields
                    )
                    if not changed:
                        present.add(previous["content_hash"])
                        continue
                added += 1
            except sqlite3.IntegrityError:
                pass
        removed = await asyncio.to_thread(db.remove_missing_archive, present)
        log.info("Archive indexed: %d new, %d removed, %d files in total", added, removed, len(files))
    except Exception:
        log.exception("Archive import failed")
