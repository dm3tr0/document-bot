from aiogram import Router

from . import chat, documents, start


def get_routers() -> Router:
    root = Router()
    root.include_router(start.router)
    # Name collection must handle its text replies before the general chat route.
    root.include_router(documents.router)
    root.include_router(chat.router)
    return root
