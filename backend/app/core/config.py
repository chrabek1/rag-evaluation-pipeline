from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_service_url: str
    database_url: str
    corpus_path: Path
    golden_dataset_path: Path
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    
settings = Settings()