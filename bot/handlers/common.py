"""Shared Telegram response helpers."""

from aiogram.types import Message


async def answer_long(message: Message, text: str, limit: int = 4000) -> None:
    """Send text in several messages if it exceeds Telegram's length limit."""
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        await message.answer(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        await message.answer(text)
