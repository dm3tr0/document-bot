import asyncio
import logging
from contextlib import suppress

from aiogram import Bot, Dispatcher

from .config import settings
from .handlers import get_routers
from .services import db, ingest, llm


async def main():
    logging.basicConfig(level=logging.INFO)
    await asyncio.to_thread(db.init_db)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()
    dp.include_router(get_routers())

    # Import the starting archive in the background: the bot answers right away,
    # and documents become searchable as soon as each one is processed.
    archive_task = None
    if settings.import_archive:
        archive_task = asyncio.create_task(ingest.import_archive())
    else:
        logging.getLogger(__name__).info("Archive import disabled (IMPORT_ARCHIVE=false)")
    try:
        await dp.start_polling(bot)
    finally:
        if archive_task is not None:
            archive_task.cancel()
            with suppress(asyncio.CancelledError):
                await archive_task
        await llm.client.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
