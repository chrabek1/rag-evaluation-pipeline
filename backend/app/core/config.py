from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_service_url: str
    database_url: str
    corpus_path: Path
    golden_dataset_path: Path

    generation_provider: Literal[
        "gemini",
        "ollama",
    ]
    generation_model: str
    generation_temperature: float = 0.0

    evaluation_provider: Literal[
        "gemini",
        "ollama",
    ]
    evaluation_model: str
    evaluation_temperature: float = 0.0

    gemini_api_key: SecretStr | None = None
    ollama_base_url: str = ("http://host.docker.internal:11434")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
