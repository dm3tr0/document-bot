import asyncio
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from ..formatting import format_sender, format_time
from ..services import db, ingest, knowledge
from ..services.extract import ALLOWED_EXTENSIONS
from .common import answer_long

router = Router()

MAX_SIZE = 20 * 1024 * 1024  # Telegram Bot API limit for bot downloads
MIME_EXTENSIONS = {
    "text/plain": ".txt",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


class AskRealName(StatesGroup):
    waiting = State()


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

    user = message.from_user
    real_name = await asyncio.to_thread(db.get_real_name, user.id)
    if real_name is None:
        # Keep Telegram file metadata, not bytes, while waiting for the name.
        await state.set_state(AskRealName.waiting)
        await state.set_data({
            "file_id": doc.file_id,
            "filename": name or f"{doc.file_unique_id}{ext}",
            "uploaded_at": message.date.isoformat(),
        })
        await message.answer("Як вас звати насправді? Напишіть ваше справжнє ім’я — я запам’ятаю його для підпису документів. Документ уже очікує на обробку.")
        return
    await process_document(message, bot, doc.file_id,
                           name or f"{doc.file_unique_id}{ext}", real_name,
                           message.date.isoformat())


@router.message(AskRealName.waiting, F.text, ~F.text.startswith("/"))
async def got_real_name(message: Message, bot: Bot, state: FSMContext):
    real_name = " ".join(message.text.split())
    if not 2 <= len(real_name) <= 100 or real_name.startswith("@"):
        await message.answer("Напишіть ваше справжнє ім’я (від 2 до 100 символів), а не Telegram-логін.")
        return
    pending = await state.get_data()
    await asyncio.to_thread(db.save_real_name, message.from_user.id, real_name)
    await state.clear()
    await message.answer(f"Дякую, {real_name}! Ім’я збережено. Обробляю документ.")
    await process_document(message, bot, pending["file_id"],
                           pending["filename"], real_name, pending["uploaded_at"])


async def process_document(
    message: Message, bot: Bot, file_id: str, filename: str,
    real_name: str, uploaded_at: str,
) -> None:
    await bot.send_chat_action(message.chat.id, "typing")
    data = (await bot.download(file_id)).read()
    user = message.from_user
    sender = ingest.Sender(id=user.id, name=real_name, username=user.username,
                           uploaded_at=uploaded_at)

    try:
        result = await ingest.ingest(data, filename, sender)
    except ingest.IngestError as exc:
        await message.answer(str(exc))
        return

    if result.duplicate:
        await message.answer(f"Документ «{result.title}» уже є в базі. Дату надсилання оновлено.")
        return

    reply = f"Збережено: {result.filename}"
    if result.updated:
        reply += "\nПравила оновлено. Попередній зміст документа більше не використовується."
    if result.summary:
        reply += f"\nКоротко: {result.summary}"
    reply += f"\nВід: {format_sender(sender.name, sender.username)}"
    reply += f"\nКоли: {format_time(uploaded_at)}"

    reply += f"\nКатегорія: {result.category}"
    await answer_long(message, reply)


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
