from datetime import datetime

import re

from pydantic import BaseModel, Field, field_validator, model_validator

_ORIGIN_RE = re.compile(r"^https?://[^\s/]+$")


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


def _validate_allowed_origins(origins: list[str] | None) -> list[str] | None:
    if origins is None:
        return None
    for origin in origins:
        if not _ORIGIN_RE.match(origin):
            raise ValueError(f"Invalid origin '{origin}': must be http(s)://hostname (no path or trailing slash)")
    return origins


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


class _RagModelFields(BaseModel):
    """Shared fields for create/update. All optional with validation constraints.

    RagModelCreate and RagModelUpdate inherit from this so new fields
    only need to be added in one place.
    """
    description: str | None = Field(None, max_length=1000)
    system_prompt: str | None = Field(None, max_length=10000)
    chat_theme: ChatTheme | None = None
    chunk_size: int | None = Field(None, ge=100, le=10000)
    chunk_overlap: int | None = Field(None, ge=0, le=2000)
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1, le=100)
    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_enabled: bool | None = None
    rerank_model: str | None = None
    rerank_candidates: int | None = Field(None, ge=1, le=500)
    rerank_threshold: float | None = Field(None, ge=0.0, le=1.0)
    keyword_search_enabled: bool | None = None
    sample_questions: list[str] | None = None
    history_turns: int | None = Field(None, ge=0, le=50)
    hosted_chat: bool | None = None
    allowed_origins: list[str] | None = None
    budget_limit: float | None = Field(None, ge=0.0)
    custom_anthropic_key: str | None = None
    custom_voyage_key: str | None = None

    @model_validator(mode="after")
    def validate_models(self):
        _validate_embedding_model(self.embedding_model)
        _validate_generation_model(self.generation_model)
        _validate_allowed_origins(self.allowed_origins)
        return self


class RagModelCreate(_RagModelFields):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9][a-z0-9-]*$")
    # Override defaults for create — empty string instead of None
    description: str = Field("", max_length=1000)
    system_prompt: str = Field("", max_length=10000)


class RagModelUpdate(_RagModelFields):
    name: str | None = Field(None, min_length=1, max_length=255)
    is_active: bool | None = None


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
