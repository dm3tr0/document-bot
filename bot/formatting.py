"""Consistent document attribution and local timestamps for bot messages."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    LOCAL_TZ = ZoneInfo("Europe/Kyiv")
except ZoneInfoNotFoundError:
    LOCAL_TZ = timezone.utc


def format_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(LOCAL_TZ).strftime("%d.%m.%Y %H:%M")


def format_sender(name: str, username: str | None) -> str:
    name = " ".join(name.split())
    return f"{name} (@{username.lstrip('@')})" if username else name


def format_source(document) -> str:
    title = " ".join((document["title"] or document["filename"]).split())
    if document["from_archive"]:
        attribution = "архів"
    else:
        sender = format_sender(document["sender_name"], document["sender_username"])
        attribution = f"{sender}, надіслано {format_time(document['uploaded_at'])}"
    return f"з «{title}» — {attribution}"
