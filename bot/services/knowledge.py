import math
import re
from collections import Counter

from ..formatting import format_source
from . import db

ONLY_OWN_DOCUMENTS = False  # False: shared knowledge base for everyone; True: archive + the user's own files
CHUNK_CHARS = 800
TOP_K = 6

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
    """Return up to top_k (source citation, text chunk) pairs, best match first.

    Blocking (reads the database): call it with asyncio.to_thread from handlers.
    """
    query_words = list(dict.fromkeys(tokenize(query)))
    if not query_words:
        return []

    docs = []  # (document id, source, context, local match, word counts)
    for row in db.list_documents(user_id):
        title = " ".join((row["title"] or row["filename"]).split())
        source = format_source(row)
        localities = re.findall(r'рукавич\w*\s+[«"]([^»"]+)[»"]', row["text"], re.IGNORECASE)
        local_match = any(location.casefold() in query.casefold() for location in localities)
        for chunk in chunk_text(row["text"]):
            context = f"Назва документа: {title}\nКатегорія: {row['category']}\nУривок:\n{chunk}"
            docs.append((row["id"], source, context, local_match, Counter(tokenize(
                f"{title}\n{row['category']}\n{chunk}"
            ))))
    if not docs:
        return []

    # Rank passages by matching word frequency, discounting common query terms.
    tf = [
        [sum(c for w, c in counts.items() if words_match(q, w)) for q in query_words]
        for _, _, _, _, counts in docs
    ]
    n = len(docs)
    idf = [
        math.log(1 + n / (1 + sum(1 for row in tf if row[i] > 0)))
        for i in range(len(query_words))
    ]

    scored = []
    for (doc_id, source, chunk, local_match, _), row in zip(docs, tf):
        score = sum(idf[i] * (1 + math.log(t)) for i, t in enumerate(row) if t > 0)
        if score > 0:
            scored.append((local_match, score, doc_id, source, chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Include the best passage from each relevant document before taking extra
    # passages from the same one, so one file cannot crowd out other sources.
    selected, extra, seen = [], [], set()
    for _, _, doc_id, source, chunk in scored:
        if doc_id in seen:
            extra.append((source, chunk))
        else:
            seen.add(doc_id)
            selected.append((source, chunk))
    return (selected + extra)[:max(0, top_k)]
