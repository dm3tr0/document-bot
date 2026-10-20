# DocumentBot

Telegram bot for Ukrainian document questions and answers, with source citations,
document updates, and optional archive indexing.

## Run with Docker

1. Install and start Docker Desktop (or a Docker Engine with Compose).
2. Create `.env` from `.env.example` and fill in `BOT_TOKEN` and `LLM_API_KEY`.
   If you already have `.env`, keep it and your existing settings.
3. Stop any locally running instance of this bot. Only one instance should poll
   Telegram with the same token.
4. Start from the project directory:

   ```sh
   docker compose up --build -d
   ```

The first run builds the image and installs Python dependencies. Later runs reuse
the image and dependency cache where possible. No inbound ports are required.

View logs:

```sh
docker compose logs -f bot
```

Stop the bot:

```sh
docker compose down
```

After changing `.env` or application code, run `docker compose up --build -d` again.

## Data and archive

- Docker keeps its database at `/data/documents.db` and uploads at `/data/storage`
  in the persistent `bot-data` volume. Ordinary restarts, rebuilds, and
  `docker compose down` preserve that data. `docker compose down -v` deletes it.
- Docker starts with its own database. Your existing project-root `documents.db`
  and local `storage/` are not imported or modified.
- The local `archive/` directory is mounted read-only at `/app/archive`. With
  `IMPORT_ARCHIVE=true`, the bot indexes it in the background at startup.
  Keep `IMPORT_ARCHIVE=false` while testing to avoid archive AI requests.
- Credentials, local databases, uploaded files, archives, and the local virtual
  environment are excluded from the image. `.env` supplies configuration at runtime.
- Containers run as an unprivileged user. The image includes timezone data for
  Ukrainian submission dates and the Python dependencies for PDF rendering.

## Run locally

With dependencies installed in your virtual environment and `.env` configured:

```sh
python -m bot
```

Local execution keeps the existing defaults: `documents.db` in the project root,
uploads in `storage/`, and archive documents in `archive/`.
