import asyncio
import hashlib
import logging
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openai import APIError, RateLimitError

from ..config import settings
from . import db, llm
from .db import UNSORTED, WHOLE_FOREST
from .extract import ALLOWED_EXTENSIONS, OCRUnavailableError, extract_document

log = logging.getLogger(__name__)

# What the AI (or a user) may answer when the forest part is not clear.
UNKNOWN_WORDS = {"unknown", "невідомо", "не визначено", "не зрозуміло", "без категорії", "none", "n/a", "-"}


class IngestError(Exception):
    """Carries a user-facing message (in Ukrainian)."""


@dataclass
class Sender:
    id: int | None
    name: str
    username: str | None = None


@dataclass
class Result:
    doc_id: int
    title: str
    filename: str
    category: str  # a category name, or WHOLE_FOREST when it still has to be asked
    summary: str
    duplicate: bool = False
    ai_ok: bool = True

    @property
    def needs_category(self) -> bool:
        return self.category == UNSORTED


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


def category_dir(category: str) -> Path:
    return settings.storage_dir / clean_title(category)


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


def move_file(src: Path, folder: Path, title: str, ext: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while _free_path(folder, title, ext, n).exists():
        n += 1
    return Path(shutil.move(src, _free_path(folder, title, ext, n)))


def _duplicate(row: sqlite3.Row) -> Result:
    return Result(row["id"], row["title"], row["filename"], row["category"], row["summary"], duplicate=True)


async def ingest(data: bytes, original_name: str, sender: Sender) -> Result:
    """Read an uploaded document, let the AI pick its forest part and title, save it, record who/when.

    General documents, unclear locations and unavailable classification use WHOLE_FOREST.
    """
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError("Підтримуються лише файли .txt, .pdf та .docx.")

    content_hash = hashlib.sha256(data).hexdigest()
    existing_doc = await asyncio.to_thread(db.find_by_hash, content_hash)
    if existing_doc:
        return _duplicate(existing_doc)

    try:
        text = await extract_document(data, ext)
    except OCRUnavailableError as exc:
        raise IngestError("Не налаштовано розпізнавання сканів. Зверніться до адміністратора.") from exc
    except (APIError, llm.TranscriptionError) as exc:
        log.warning("Document vision extraction failed: %s", type(exc).__name__)
        raise IngestError("\u041d\u0435 \u0432\u0434\u0430\u043b\u043e\u0441\u044f \u0440\u043e\u0437\u043f\u0456\u0437\u043d\u0430\u0442\u0438 \u0441\u043a\u0430\u043d. \u0421\u043f\u0440\u043e\u0431\u0443\u0439\u0442\u0435 \u043f\u0456\u0437\u043d\u0456\u0448\u0435.") from exc
    except Exception as exc:  # corrupt, encrypted, not really a pdf/docx...
        raise IngestError("Не вдалося прочитати файл. Можливо, він пошкоджений або захищений паролем.") from exc
    if not text.strip():
        raise IngestError("У файлі немає тексту, який я можу прочитати.")

    categories = await asyncio.to_thread(db.list_categories)
    ai_ok = True
    try:
        raw_category, ai_title, summary = await llm.classify(text, categories, filename=original_name)
    except (APIError, llm.ClassificationError):
        log.warning("Document analysis failed; saved in whole-forest category")
        raw_category, ai_title, summary, ai_ok = "", "", "", False
    category = normalize_category(raw_category, categories) or WHOLE_FOREST

    first_line = next((line for line in text.splitlines() if line.strip()), "")
    stem = clean_title(Path(original_name).stem)
    title = clean_title(ai_title) or stem or clean_title(first_line) or "Документ"

    path = await asyncio.to_thread(save_file, category_dir(category), title, ext, data)
    try:
        doc_id = await asyncio.to_thread(
            db.add_document,
            title=title, filename=path.name, category=category, summary=summary, text=text,
            content_hash=content_hash, sender_id=sender.id, sender_name=sender.name,
            sender_username=sender.username, from_archive=False,
        )
    except sqlite3.IntegrityError:  # the very same file was added a moment ago
        path.unlink(missing_ok=True)
        return _duplicate(await asyncio.to_thread(db.find_by_hash, content_hash))
    return Result(doc_id, title, path.name, category, summary, ai_ok=ai_ok)


async def assign_category(doc_id: int, raw: str) -> str | None:
    """Put a document into the category named by the user. Returns the final name, None if unusable."""
    existing = await asyncio.to_thread(db.list_categories)
    category = normalize_category(raw, existing)
    row = await asyncio.to_thread(db.get_document, doc_id)
    if category is None or row is None:
        return None
    filename = row["filename"]
    src = category_dir(row["category"]) / filename
    if not row["from_archive"] and src.exists():
        dst = await asyncio.to_thread(
            move_file, src, category_dir(category), row["title"], Path(filename).suffix
        )
        filename = dst.name
    await asyncio.to_thread(db.update_category, doc_id, category, filename)
    return category


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
                await asyncio.to_thread(
                    db.add_document,
                    title=path.stem, filename=path.name, category=category, summary=summary, text=text,
                    content_hash=content_hash, sender_id=None, sender_name=db.ARCHIVE,
                    sender_username=None, from_archive=True,
                )
                added += 1
            except sqlite3.IntegrityError:
                pass
        removed = await asyncio.to_thread(db.remove_missing_archive, present)
        log.info("Archive indexed: %d new, %d removed, %d files in total", added, removed, len(files))
    except Exception:
        log.exception("Archive import failed")
