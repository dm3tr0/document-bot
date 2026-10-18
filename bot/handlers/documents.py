import asyncio
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from ..services import db, ingest, knowledge
from ..services.extract import ALLOWED_EXTENSIONS

router = Router()

MAX_SIZE = 20 * 1024 * 1024  # Telegram Bot API limit for bot downloads
MAX_CATEGORY_LEN = 40
MIME_EXTENSIONS = {
    "text/plain": ".txt",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

try:
    LOCAL_TZ = ZoneInfo("Europe/Kyiv")
except ZoneInfoNotFoundError:  # on Windows this needs: pip install tzdata
    LOCAL_TZ = timezone.utc


class AskCategory(StatesGroup):
    waiting = State()  # the bot asked which forest part a document is for


def format_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(LOCAL_TZ).strftime("%d.%m.%Y %H:%M")


def format_sender(name: str, username: str | None) -> str:
    return f"{name} (@{username})" if username else name


async def answer_long(message: Message, text: str, limit: int = 4000) -> None:
    """Send text in several messages if it exceeds Telegram's length limit."""
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        await message.answer(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        await message.answer(text)


@router.message(F.document)
async def handle_document(message: Message, bot: Bot, state: FSMContext):
    doc = message.document
    name = doc.file_name or ""
    ext = Path(name).suffix.lower() or MIME_EXTENSIONS.get(doc.mime_type or "", "")

    if ext not in ALLOWED_EXTENSIONS:
        await message.answer("Надішліть, будь ласка, файл у форматі .txt, .pdf або .docx.")
        return
    if doc.file_size and doc.file_size > MAX_SIZE:
        await message.answer("Файл завеликий. Ліміт — 20 МБ.")
        return

    await bot.send_chat_action(message.chat.id, "typing")
    data = (await bot.download(doc)).read()
    user = message.from_user
    sender = ingest.Sender(id=user.id, name=user.full_name, username=user.username)

    try:
        result = await ingest.ingest(data, name or f"{doc.file_unique_id}{ext}", sender)
    except ingest.IngestError as exc:
        await message.answer(str(exc))
        return

    if result.duplicate:
        await message.answer(f"Такий документ уже є в базі: «{result.title}» ({result.category}).")
        return

    reply = f"Збережено: {result.filename}"
    if result.summary:
        reply += f"\nКоротко: {result.summary}"
    reply += f"\nВід: {format_sender(sender.name, sender.username)}"
    reply += f"\nКоли: {datetime.now(LOCAL_TZ).strftime('%d.%m.%Y %H:%M')}"

    if not result.needs_category:
        reply += f"\nЧастина лісу: {result.category}"
        await answer_long(message, reply)
        return

    # The AI could not tell which forest part the advice is for (or was unavailable): ask the user.
    existing = await asyncio.to_thread(db.list_categories)
    reply += "\n\nНе можу зрозуміти з тексту, для якої частини лісу ця порада."
    if not result.ai_ok:
        reply += " (Аналіз ШІ зараз недоступний.)"
    reply += "\nНапишіть назву частини лісу, наприклад «Хвойний ліс»."
    if existing:
        reply += f"\nУже є: {', '.join(existing)}."
    reply += "\nАбо /skip, щоб залишити без категорії."
    await state.set_state(AskCategory.waiting)
    await state.update_data(doc_id=result.doc_id)
    await answer_long(message, reply)


@router.message(AskCategory.waiting, F.text, ~F.text.startswith("/"))
async def got_category(message: Message, state: FSMContext):
    text = message.text.strip()
    if len(text) > MAX_CATEGORY_LEN or "?" in text:
        await message.answer(
            f"Напишіть коротку назву частини лісу (до {MAX_CATEGORY_LEN} символів), "
            "наприклад «Болото», або /skip."
        )
        return
    doc_id = (await state.get_data())["doc_id"]
    category = await ingest.assign_category(doc_id, text)
    if category is None:
        await message.answer("Не вдалося використати таку назву. Спробуйте ще раз або /skip.")
        return
    await state.clear()
    await message.answer(f"Готово. Документ тепер у категорії «{category}».")


@router.message(AskCategory.waiting, Command("skip"))
async def skip_category(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(f"Добре, документ залишається в категорії «{db.UNSORTED}».")


@router.message(Command("docs"))
async def cmd_docs(message: Message):
    user_id = message.from_user.id if knowledge.ONLY_OWN_DOCUMENTS else None
    rows = await asyncio.to_thread(db.list_documents, user_id)

    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    # alphabetical, with the "no category" group last
    order = sorted(groups, key=lambda c: (c == db.UNSORTED, c.casefold()))

    lines = []
    for category in order:
        lines.append(f"{category} ({len(groups[category])}):")
        for row in groups[category]:
            who = format_sender(row["sender_name"], row["sender_username"])
            lines.append(f"  • {row['filename']} — {who}, {format_time(row['uploaded_at'])}")
        lines.append("")
    await answer_long(message, "\n".join(lines).strip() or "Документів ще немає. Надішліть мені файл.")