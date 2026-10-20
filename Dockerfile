FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/data/documents.db \
    STORAGE_DIR=/data/storage \
    ARCHIVE_DIR=/app/archive

WORKDIR /app

# Timezone data is needed to display submission dates in Europe/Kyiv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 bot \
    && useradd --uid 10001 --gid bot --no-create-home bot \
    && mkdir -p /data/storage /app/archive \
    && chown -R bot:bot /data

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/

USER bot
CMD ["python", "-m", "bot"]
