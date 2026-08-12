from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_service_url: str
    database_url: str
    corpus_path: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    
settings = Settings()