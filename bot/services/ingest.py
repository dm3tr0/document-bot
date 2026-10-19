import asyncio
import hashlib
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import APIError, RateLimitError

from ..config import settings
from . import db, llm
from .db import WHOLE_FOREST
from .extract import ALLOWED_EXTENSIONS, OCRUnavailableError, extract_document
from .knowledge import chunk_text, tokenize, words_match

log = logging.getLogger(__name__)

UPDATE_CANDIDATES = 8
NEW_DOCUMENT_CHARS = 3000
CANDIDATE_CHARS = 800

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
    name = clean_title(raw, limit=120)
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


def comparison_excerpt(text: str, query_words: set[str], limit: int) -> str:
    """Retain scope near the heading, relevant advice, and the ending within a cap."""
    if len(text) <= limit:
        return text
    head_size, tail_size = limit // 3, limit // 6
    middle = text[head_size:-tail_size]
    passages = chunk_text(middle, size=max(100, limit // 2))
    def relevance(passage):
        return sum(any(words_match(word, query) for query in query_words)
                   for word in set(tokenize(passage)))
    best = max(passages, key=relevance, default="")
    separator = "\n[...]\n"
    remaining = limit - head_size - tail_size - 2 * len(separator)
    return text[:head_size] + separator + best[:remaining] + separator + text[-tail_size:]


def prepare_analysis(text: str, documents: list) -> tuple[str, list[dict]]:
    """Bound request size while preserving candidate IDs for safe updates."""
    sample = comparison_excerpt(text, set(tokenize(text[:500])), NEW_DOCUMENT_CHARS)
    query_words = set(tokenize(sample))
    def similarity(row):
        body_words = set(tokenize(row["text"]))
        return sum(1 for word in body_words
                   if any(words_match(word, query) for query in query_words))
    ranked = sorted(((similarity(row), row) for row in documents),
                    key=lambda item: item[0], reverse=True)
    candidates = [
        {"id": row["id"], "категорія": row["category"],
         "опис": row["summary"][:240],
         "текст": comparison_excerpt(row["text"], query_words, CANDIDATE_CHARS)}
        for score, row in ranked[:UPDATE_CANDIDATES] if score > 0
    ]
    return sample, candidates


async def analyze_document(text: str) -> tuple[llm.DocumentAnalysis, sqlite3.Row | None]:
    documents = await asyncio.to_thread(db.list_documents)
    categories = await asyncio.to_thread(db.list_categories)
    sample, candidates = await asyncio.to_thread(prepare_analysis, text, documents)
    started_at = time.monotonic()
    analysis = await llm.analyze_document(sample, categories, candidates)
    log.info("Document analysis candidates=%d context_chars=%d elapsed_seconds=%.2f",
             len(candidates), len(sample) + sum(len(row["текст"]) for row in candidates),
             time.monotonic() - started_at)
    previous = next((row for row in documents if row["id"] == analysis.update_id), None)
    return analysis, previous


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

    try:
        analysis, previous = await analyze_document(text)
    except (APIError, llm.ClassificationError) as exc:
        # Do not insert an unverified second copy if the combined request fails.
        raise IngestError("Не вдалося проаналізувати документ і перевірити оновлення правил. Спробуйте пізніше.") from exc
    categories = await asyncio.to_thread(db.list_categories)
    category = normalize_category(analysis.category, categories) or WHOLE_FOREST
    ai_title, summary = analysis.title, analysis.summary

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

            try:
                analysis, previous = await analyze_document(text)
                if previous is not None:
                    # An old archive file must not reintroduce replaced rules at restart.
                    if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) <= datetime.fromisoformat(previous["uploaded_at"]):
                        if previous["from_archive"]:
                            present.add(previous["content_hash"])
                        continue
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
            categories = await asyncio.to_thread(db.list_categories)
            category = normalize_category(analysis.category, categories) or WHOLE_FOREST
            summary = analysis.summary

            try:
                fields = dict(
                    title=clean_title(analysis.title) or "Документ", filename=path.name, category=category, summary=summary, text=text,
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
