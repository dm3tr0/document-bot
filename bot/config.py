from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent  # project root, next to the bot/ package


class Settings(BaseSettings):
    bot_token: str
    llm_api_key: str
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str
    llm_requests_per_minute: int = Field(default=20, gt=0)
    llm_tokens_per_minute: int = Field(default=6000, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_vision_model: str
    import_archive: bool = False  # disable background archive processing during testing
    storage_dir: Path = ROOT / "storage"  # all uploaded documents
    archive_dir: Path = ROOT / "archive"  # starting archive, imported on startup
    db_path: Path = ROOT / "documents.db"  # who/when/category for every document
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8")


settings = Settings()
