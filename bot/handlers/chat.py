import asyncio
import logging

from aiogram import F, Router
from aiogram.types import Message
from openai import APIError

from ..services import knowledge, llm
from .common import answer_long

router = Router()
log = logging.getLogger(__name__)


@router.message(F.text, ~F.text.startswith("/"))
async def chat(message: Message):
    # Text messages are questions only; document ingestion requires a file upload.
    await message.bot.send_chat_action(message.chat.id, "typing")

    user_id = message.from_user.id if knowledge.ONLY_OWN_DOCUMENTS else None
    # Database reads and text ranking run in a worker so Telegram stays responsive.
    excerpts = await asyncio.to_thread(knowledge.search, message.text, user_id)
    if not excerpts:
        await message.answer(llm.NO_INFO)  # nothing relevant found: don't even ask the model
        return

    try:
        answer = await llm.answer_from_documents(message.text, excerpts)
    except llm.AnswerError:
        log.warning("LLM answer had invalid structure or source references")
        await message.answer("Не вдалося сформувати відповідь із підтвердженими джерелами. Спробуйте уточнити питання.")
        return
    except APIError:
        log.exception("LLM request failed")
        await message.answer("Сервіс ШІ зараз недоступний. Спробуйте, будь ласка, пізніше.")
        return
    await answer_long(message, answer)
