from datetime import datetime

from pydantic import BaseModel, Field


class RagModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str = ""
    greeting: str = ""
    placeholder: str = ""
    system_prompt: str = ""
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    similarity_threshold: float | None = None
    top_k: int | None = None
    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_enabled: bool | None = None
    rerank_model: str | None = None
    public_access: bool | None = None
    allowed_origins: list[str] | None = None
    budget_limit: float | None = None


class RagModelUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    greeting: str | None = None
    placeholder: str | None = None
    system_prompt: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    similarity_threshold: float | None = None
    top_k: int | None = None
    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_enabled: bool | None = None
    rerank_model: str | None = None
    public_access: bool | None = None
    allowed_origins: list[str] | None = None
    budget_limit: float | None = None
    is_active: bool | None = None


class RagModelPublic(BaseModel):
    name: str
    slug: str
    description: str
    greeting: str
    placeholder: str
    allowed_origins: list[str]
    public_access: bool
    accepting_requests: bool = True

    model_config = {"from_attributes": True}


class RagModelRead(BaseModel):
    id: int
    name: str
    slug: str
    description: str
    greeting: str
    placeholder: str
    system_prompt: str
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    top_k: int
    embedding_model: str
    generation_model: str
    reranker_enabled: bool
    rerank_model: str
    public_access: bool
    allowed_origins: list[str]
    budget_limit: float
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
