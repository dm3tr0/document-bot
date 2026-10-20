from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

router = Router()

INTRO = (
    "Вітаю! Я лісовий помічник!\n\n"
    "Надішліть мені текстовий файл із лісівничими порадами, і я визначу, "
    "якої частини лісу він стосується, та збережу його.\n"
    "Документи приймаю лише файлами .txt, .pdf або .docx; фото не приймаю.\n"
    "Звичайні текстові повідомлення сприймаю лише як питання, а не документи для збереження.\n\n"
    "Команди:\n"
    "/docs — документи за категоріями\n"
    "/help — допомога"
)


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(INTRO)


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(INTRO)
