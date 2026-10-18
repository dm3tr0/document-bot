import math
import re
from collections import Counter

from . import db

ONLY_OWN_DOCUMENTS = False  # False: shared knowledge base for everyone; True: archive + the user's own files
CHUNK_CHARS = 800
TOP_K = 4

STOPWORDS = {
    "для", "при", "про", "від", "яка", "який", "які", "яке", "або", "але", "цей", "ця",
    "це", "ці", "коли", "чому", "чим", "щоб", "його", "вона", "вони", "наш", "ваш",
    "треба", "можна", "якщо", "також", "після", "перед", "між", "через", "як", "що",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[\w'’]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def words_match(a: str, b: str) -> bool:
    """Cheap Ukrainian 'stemming': words match if they share a long enough common prefix."""
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    shortest = min(len(a), len(b))
    needed = 3 if shortest <= 4 else max(4, shortest - 2)
    return common >= min(needed, shortest)


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    chunks, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        while len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            cut = max(para.rfind("\n", 0, size), para.rfind(" ", 0, size))
            cut = cut if cut > size // 2 else size  # prefer cutting at a line break or space
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > size:
            if current:
                chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def search(query: str, user_id: int | None = None, top_k: int = TOP_K) -> list[tuple[str, str]]:
    """Return up to top_k (source file name, text chunk) pairs, best match first.

    Blocking (reads the database): call it with asyncio.to_thread from handlers.
    """
    query_words = list(dict.fromkeys(tokenize(query)))
    if not query_words:
        return []

    docs = []  # (source, chunk, Counter of words)
    for row in db.list_documents(user_id):
        for chunk in chunk_text(row["text"]):
            docs.append((row["filename"], chunk, Counter(tokenize(chunk))))
    if not docs:
        return []

    tf = [
        [sum(c for w, c in counts.items() if words_match(q, w)) for q in query_words]
        for _, _, counts in docs
    ]
    n = len(docs)
    idf = [
        math.log(1 + n / (1 + sum(1 for row in tf if row[i] > 0)))
        for i in range(len(query_words))
    ]

    scored = []
    for (source, chunk, _), row in zip(docs, tf):
        score = sum(idf[i] * (1 + math.log(t)) for i, t in enumerate(row) if t > 0)
        if score > 0:
            scored.append((score, source, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(source, chunk) for _, source, chunk in scored[:top_k]]