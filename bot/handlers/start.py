from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

router = Router()

INTRO = (
    "Вітаю! Я лісовий помічник!\n\n"
    "Надішліть мені текстовий файл із лісівничими порадами, і я визначу, "
    "якої частини лісу він стосується, та збережу його.\n"
    "Також можете просто поставити запитання про ліс.\n\n"
    "Команди:\n"
    "/docs — ваші документи за категоріями\n"
    "/reset — очистити розмову"
)


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(INTRO)


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(INTRO)