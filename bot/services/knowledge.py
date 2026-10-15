import math
import re
from collections import Counter
from pathlib import Path

SAVE_DIR = Path("storage")  # temporary storage ill make db soon
ONLY_OWN_DOCUMENTS = False  # False: shared knowledge base for everyone; True: each user sees only their files
CHUNK_CHARS = 800
TOP_K = 4

STOPWORDS = {
    "для", "при", "про", "від", "яка", "який", "які", "яке", "або", "але", "цей", "ця",
    "це", "ці", "коли", "чому", "чим", "щоб", "його", "вона", "вони", "наш", "ваш",
    "треба", "можна", "якщо", "також", "після", "перед", "між", "через", "як", "що",
}


def decode_text(data: bytes) -> str:
    """UTF-8 first (with BOM support), then Windows-1251, common for older Ukrainian files."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")


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
            chunks.append(para[:size])
            para = para[size:]
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

    Blocking (reads files from disk): call it with asyncio.to_thread from handlers.
    """
    query_words = list(dict.fromkeys(tokenize(query)))
    if not query_words or not SAVE_DIR.exists():
        return []

    docs = []  # (source, chunk, Counter of words)
    for path in SAVE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if user_id is not None and path.parent.name != str(user_id):
            continue
        source = re.sub(r"^\d+_", "", path.name)  # drop the timestamp prefix
        for chunk in chunk_text(decode_text(path.read_bytes())):
            docs.append((source, chunk, Counter(tokenize(chunk))))
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