import asyncio
import logging

from aiogram import Bot, Dispatcher

from .config import settings
from .handlers import get_routers

async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()
    dp.include_router(get_routers())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())