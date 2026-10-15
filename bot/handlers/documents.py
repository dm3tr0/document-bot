import logging
import re
import time
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import Message
from openai import APIError

from ..services import llm
from ..services.knowledge import decode_text

router = Router()
log = logging.getLogger(__name__)

SAVE_DIR = Path("storage")
MAX_SIZE = 20 * 1024 * 1024  # Telegram Bot API limit for bot downloads
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}

# id (used as folder name) -> label shown to users. Keep "other" LAST: it's the fallback.
CATEGORIES = {
    "trees": "Дерева та деревостан",
    "undergrowth": "Підлісок і трав'яний покрив",
    "soil": "Ґрунт і лісова підстилка",
    "water": "Водні об'єкти та водний режим",
    "wildlife": "Тварини та біорізноманіття",
    "fire": "Пожежна безпека",
    "pests": "Шкідники та хвороби лісу",
    "reforestation": "Лісовідновлення та садіння",
    "other": "Інше",
}


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]", "_", Path(name).name)


def is_text_document(doc) -> bool:
    ext = Path(doc.file_name or "").suffix.lower()
    return (doc.mime_type or "").startswith("text/") or ext in TEXT_EXTENSIONS


@router.message(F.document)
async def handle_document(message: Message, bot: Bot):
    doc = message.document

    if not is_text_document(doc):
        await message.answer("Надішліть, будь ласка, текстовий файл (.txt, .md, .csv, .json, .log).")
        return
    if doc.file_size and doc.file_size > MAX_SIZE:
        await message.answer("Файл завеликий. Ліміт — 20 МБ.")
        return

    await bot.send_chat_action(message.chat.id, "typing")
    data = (await bot.download(doc)).read()
    text = decode_text(data)
    if not text.strip():
        await message.answer("Файл порожній.")
        return

    note = ""
    summary = ""
    try:
        category, summary = await llm.classify(text, CATEGORIES)
    except APIError:
        log.exception("Document classification failed")
        category = list(CATEGORIES)[-1]
        note = "\n(Аналіз ШІ недоступний, тому документ збережено в категорії за замовчуванням.)"

    target_dir = SAVE_DIR / category / str(message.from_user.id)
    target_dir.mkdir(parents=True, exist_ok=True)
    original = doc.file_name or f"{doc.file_unique_id}.txt"
    filename = f"{int(time.time())}_{safe_name(original)}"
    (target_dir / filename).write_bytes(data)

    reply = f"Частина лісу: {CATEGORIES[category]}"
    if summary:
        reply += f"\nКоротко: {summary}"
    reply += f"\nЗбережено як: {filename}{note}"
    await message.answer(reply[:4096])


@router.message(Command("docs"))
async def cmd_docs(message: Message):
    user_id = str(message.from_user.id)
    lines = []
    for category, label in CATEGORIES.items():
        folder = SAVE_DIR / category / user_id
        files = sorted(p.name for p in folder.iterdir()) if folder.exists() else []
        if files:
            lines.append(f"{label} ({len(files)}):")
            lines += [f"  - {name}" for name in files]
    text = "\n".join(lines) if lines else "Документів ще немає. Надішліть мені текстовий файл."
    await message.answer(text[:4096])