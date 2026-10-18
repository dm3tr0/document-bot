import asyncio
import logging

from aiogram import Bot, Dispatcher

from .config import settings
from .handlers import get_routers
from .services import db, ingest


async def main():
    logging.basicConfig(level=logging.INFO)
    await asyncio.to_thread(db.init_db)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()
    dp.include_router(get_routers())

    # Import the starting archive in the background: the bot answers right away,
    # and documents become searchable as soon as each one is processed.
    archive_task = asyncio.create_task(ingest.import_archive())
    try:
        await dp.start_polling(bot)
    finally:
        archive_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())