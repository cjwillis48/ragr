from datetime import datetime

from pydantic import BaseModel, Field


class ChatTheme(BaseModel):
    label: str | None = None
    greeting: str | None = None
    placeholder: str | None = None
    launcher_hint: str | None = None
    primary_color: str | None = None
    bg_color: str | None = None
    text_color: str | None = None
    user_bubble_color: str | None = None
    bot_bubble_color: str | None = None
    font_family: str | None = None
    border_radius: int | None = None


class RagModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str = ""
    system_prompt: str = ""
    chat_theme: ChatTheme | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    similarity_threshold: float | None = None
    top_k: int | None = None
    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_enabled: bool | None = None
    rerank_model: str | None = None
    history_turns: int | None = None
    hosted_chat: bool | None = None
    allowed_origins: list[str] | None = None
    budget_limit: float | None = None
    custom_anthropic_key: str | None = None
    custom_voyage_key: str | None = None


class RagModelUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    similarity_threshold: float | None = None
    top_k: int | None = None
    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_enabled: bool | None = None
    rerank_model: str | None = None
    history_turns: int | None = None
    hosted_chat: bool | None = None
    allowed_origins: list[str] | None = None
    budget_limit: float | None = None
    is_active: bool | None = None
    custom_anthropic_key: str | None = None
    custom_voyage_key: str | None = None


class RagModelPublic(BaseModel):
    name: str
    slug: str
    description: str
    chat_theme: ChatTheme | None = None
    allowed_origins: list[str]
    hosted_chat: bool
    accepting_requests: bool = True

    model_config = {"from_attributes": True}


class RagModelRead(BaseModel):
    id: int
    name: str
    slug: str
    description: str
    system_prompt: str
    chat_theme: ChatTheme | None = None
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    top_k: int
    embedding_model: str
    generation_model: str
    reranker_enabled: bool
    rerank_model: str
    history_turns: int
    hosted_chat: bool
    allowed_origins: list[str]
    budget_limit: float
    has_custom_anthropic_key: bool = False
    has_custom_voyage_key: bool = False
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_model(cls, model) -> "RagModelRead":
        data = cls.model_validate(model)
        data.has_custom_anthropic_key = bool(model.custom_anthropic_key)
        data.has_custom_voyage_key = bool(model.custom_voyage_key)
        return data
