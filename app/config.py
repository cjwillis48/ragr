from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = ""

    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    encryption_key: str = ""

    clerk_secret_key: str = ""
    superuser_id: str = ""
    console_origins: list[str] = ["http://localhost:5173"]

    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "ragr-uploads"
    r2_presigned_expiry: int = 3600

    trusted_proxy_ips: list[str] = []
    rate_limit_per_min: int = 10
    max_upload_size_mb: int = 50
    max_upload_files: int = 20

    default_similarity_threshold: float = 0.3
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 100
    default_top_k: int = 15
    default_budget_limit: float = 10.0
    default_generation_model: str = "claude-haiku-4-5"
    default_embedding_model: str = "voyage-4-lite"
    default_rerank_model: str = "rerank-2.5-lite"
    default_reranker_enabled: bool = True
    default_history_turns: int = 10
    default_hosted_chat: bool = True
    default_max_tokens: int = 1024


settings = Settings()
