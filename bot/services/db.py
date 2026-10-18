import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from ..config import settings

WHOLE_FOREST = "Для всього лісу"
UNSORTED = "Без категорії"  # used while the forest part of a document is unknown
ARCHIVE = "Початковий архів"  # sender name of documents indexed from the archive folder
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    filename        TEXT NOT NULL,
    category        TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    text            TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,
    sender_id       INTEGER,
    sender_name     TEXT NOT NULL,
    sender_username TEXT,
    uploaded_at     TEXT NOT NULL,
    from_archive    INTEGER NOT NULL DEFAULT 0
)
"""
# Each function opens its own short-lived connection, so they are safe to call
# from worker threads via asyncio.to_thread.


def _connect() -> sqlite3.Connection:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(SCHEMA)


def find_by_hash(content_hash: str) -> sqlite3.Row | None:
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()


def add_document(
    *, title, filename, category, summary, text, content_hash,
    sender_id, sender_name, sender_username, from_archive,
) -> int:
    """Insert a document. Raises sqlite3.IntegrityError if the same content is already stored."""
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            """INSERT INTO documents (title, filename, category, summary, text, content_hash,
                   sender_id, sender_name, sender_username, uploaded_at, from_archive)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (title, filename, category, summary, text, content_hash,
             sender_id, sender_name, sender_username, uploaded_at, int(from_archive)),
        )
        return cursor.lastrowid


def list_categories(limit: int = 30) -> list[str]:
    """Categories already in use, most used first (UNSORTED is not a real category)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """SELECT category FROM documents WHERE category != ?
               GROUP BY category ORDER BY COUNT(*) DESC, category LIMIT ?""",
            (UNSORTED, limit),
        ).fetchall()
    names = [row["category"] for row in rows if row["category"] != WHOLE_FOREST]
    return ([WHOLE_FOREST] + names)[:max(0, limit)]


def list_categories(limit: int = 30) -> list[str]:
    """Categories already in use, most used first (UNSORTED is not a real category)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """SELECT category FROM documents WHERE category != ?
               GROUP BY category ORDER BY COUNT(*) DESC, category LIMIT ?""",
            (UNSORTED, limit),
        ).fetchall()
    names = [row["category"] for row in rows if row["category"] != WHOLE_FOREST]
    return ([WHOLE_FOREST] + names)[:max(0, limit)]


def get_document(doc_id: int) -> sqlite3.Row | None:
    with closing(_connect()) as conn:
        return conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()


def update_category(doc_id: int, category: str, filename: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE documents SET category = ?, filename = ? WHERE id = ?",
            (category, filename, doc_id),
        )

def remove_missing_archive(present_hashes: set[str]) -> int:
    """Drop archive rows whose file is no longer in the archive folder. Returns how many."""
    with closing(_connect()) as conn, conn:
        rows = conn.execute("SELECT id, content_hash FROM documents WHERE from_archive = 1").fetchall()
        stale = [(row["id"],) for row in rows if row["content_hash"] not in present_hashes]
        conn.executemany("DELETE FROM documents WHERE id = ?", stale)
        return len(stale)


