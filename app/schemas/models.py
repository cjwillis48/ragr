from datetime import datetime

from pydantic import BaseModel, Field, model_validator


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
    show_sample_questions_in_greeting: bool | None = None


# Supported embedding models and their vector dimensions.
# The DB column is Vector(1024), so only 1024-dim models are allowed.
SUPPORTED_EMBEDDING_MODELS: dict[str, int] = {
    "voyage-4-lite": 1024,
    "voyage-4": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
}

ALLOWED_EMBEDDING_MODELS = {k for k, v in SUPPORTED_EMBEDDING_MODELS.items() if v == 1024}

ALLOWED_GENERATION_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4",
}


def _validate_embedding_model(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    if model_name not in ALLOWED_EMBEDDING_MODELS:
        allowed = ", ".join(sorted(ALLOWED_EMBEDDING_MODELS))
        raise ValueError(
            f"Unsupported embedding model '{model_name}'. "
            f"Allowed models (1024-dim): {allowed}"
        )
    return model_name


def _validate_generation_model(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    if model_name not in ALLOWED_GENERATION_MODELS:
        allowed = ", ".join(sorted(ALLOWED_GENERATION_MODELS))
        raise ValueError(
            f"Unsupported generation model '{model_name}'. "
            f"Allowed models: {allowed}"
        )
    return model_name


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
    rerank_candidates: int | None = None
    rerank_threshold: float | None = None
    keyword_search_enabled: bool | None = None
    sample_questions: list[str] | None = None
    history_turns: int | None = None
    hosted_chat: bool | None = None
    allowed_origins: list[str] | None = None
    budget_limit: float | None = None
    custom_anthropic_key: str | None = None
    custom_voyage_key: str | None = None

    @model_validator(mode="after")
    def validate_models(self):
        _validate_embedding_model(self.embedding_model)
        _validate_generation_model(self.generation_model)
        return self


class RagModelUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str | None = None
    chat_theme: ChatTheme | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    similarity_threshold: float | None = None
    top_k: int | None = None
    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_enabled: bool | None = None
    rerank_model: str | None = None
    rerank_candidates: int | None = None
    rerank_threshold: float | None = None
    keyword_search_enabled: bool | None = None
    sample_questions: list[str] | None = None
    history_turns: int | None = None
    hosted_chat: bool | None = None
    allowed_origins: list[str] | None = None
    budget_limit: float | None = None
    is_active: bool | None = None
    custom_anthropic_key: str | None = None
    custom_voyage_key: str | None = None

    @model_validator(mode="after")
    def validate_models(self):
        _validate_embedding_model(self.embedding_model)
        _validate_generation_model(self.generation_model)
        return self


class RagModelPublic(BaseModel):
    name: str
    slug: str
    description: str
    chat_theme: ChatTheme | None = None
    allowed_origins: list[str]
    hosted_chat: bool
    sample_questions: list[str] = []
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
    rerank_candidates: int
    rerank_threshold: float
    keyword_search_enabled: bool
    sample_questions: list[str]
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
