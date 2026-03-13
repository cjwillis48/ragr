from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://ragr:ragr@localhost:5432/ragr"

    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    ragr_api_key: str = ""
    encryption_key: str = ""

    clerk_secret_key: str = ""
    superuser_id: str = ""
    console_origins: list[str] = ["http://localhost:5173"]

    rate_limit_per_min: int = 10

    default_similarity_threshold: float = 0.5
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 100
    default_top_k: int = 15
    default_budget_limit: float = 10.0
    default_generation_model: str = "claude-haiku-4-5"
    default_embedding_model: str = "voyage-4-lite"
    default_rerank_model: str = "rerank-2.5-lite"


settings = Settings()
