from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Voyage AI (Embeddings)
    voyage_api_key: str
    voyage_embedding_model: str = "voyage-3"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "digimon_knowledge"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "digimon_db"
    postgres_user: str = "digimon"
    postgres_password: str = "digimon_pass"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # DAPI
    dapi_base_url: str = "https://digi-api.com/api/v1"

    # Ingestion — rate limiting for Voyage AI free tier (3 RPM, 10K TPM)
    ingest_limit: int = 100      # 0 = all Digimon; 100 ≈ 5 min with free tier
    voyage_embed_batch_size: int = 20   # chunks per API call
    voyage_rpm: int = 3          # requests per minute (free tier = 3)

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
