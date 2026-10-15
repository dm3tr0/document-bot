import asyncio
import logging

from aiogram import F, Router
from aiogram.types import Message
from openai import APIError

from ..services import knowledge, llm

router = Router()
log = logging.getLogger(__name__)


@router.message(F.text, ~F.text.startswith("/"))
async def chat(message: Message):
    await message.bot.send_chat_action(message.chat.id, "typing")

    user_id = message.from_user.id if knowledge.ONLY_OWN_DOCUMENTS else None
    # File reading is blocking, so run it in a thread to keep the bot responsive.
    excerpts = await asyncio.to_thread(knowledge.search, message.text, user_id)
    if not excerpts:
        await message.answer(llm.NO_INFO)  # nothing relevant found: don't even ask the model
        return

    try:
        answer = await llm.answer_from_documents(message.text, excerpts)
    except APIError:
        log.exception("LLM request failed")
        await message.answer("Сервіс ШІ зараз недоступний. Спробуйте, будь ласка, пізніше.")
        return
    await message.answer(answer[:4096])  # Telegram's message limit