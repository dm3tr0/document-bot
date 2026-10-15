from aiogram import Router
from . import start, documents, chat

def get_routers() -> Router:
    root = Router()
    root.include_router(start.router)
    root.include_router(documents.router)
    root.include_router(chat.router)
    return root