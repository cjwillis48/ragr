from datetime import datetime

import re

from pydantic import BaseModel, Field, field_validator, model_validator

_ORIGIN_RE = re.compile(r"^https?://[^\s/]+$")


_HEX_COLOR = r"^#[0-9a-fA-F]{3,8}$"


class ChatTheme(BaseModel):
    label: str | None = Field(None, max_length=100)
    greeting: str | None = Field(None, max_length=500)
    placeholder: str | None = Field(None, max_length=200)
    launcher_hint: str | None = Field(None, max_length=200)
    primary_color: str | None = Field(None, pattern=_HEX_COLOR, max_length=9)
    bg_color: str | None = Field(None, pattern=_HEX_COLOR, max_length=9)
    text_color: str | None = Field(None, pattern=_HEX_COLOR, max_length=9)
    user_bubble_color: str | None = Field(None, pattern=_HEX_COLOR, max_length=9)
    bot_bubble_color: str | None = Field(None, pattern=_HEX_COLOR, max_length=9)
    font_family: str | None = Field(None, max_length=100, pattern=r"^[a-zA-Z0-9 ,'\"-]+$")
    border_radius: int | None = Field(None, ge=0, le=50)
    show_sample_questions_in_greeting: bool | None = None


SUPPORTED_EMBEDDING_MODELS: set[str] = {
    "voyage-4-lite",
    "voyage-4",
}

SUPPORTED_GENERATION_MODELS: set[str] = {
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
}


def _validate_model(allowlist: set[str], model_name: str | None):
    if model_name is None:
        return None
    if model_name not in allowlist:
        allowed = ", ".join(sorted(allowlist))
        raise ValueError(f"Unsupported model. Allowed: {allowed}")
    return model_name


def _validate_embedding_model(model_name: str | None) -> str | None:
    return _validate_model(SUPPORTED_EMBEDDING_MODELS, model_name)


def _validate_generation_model(model_name: str | None) -> str | None:
    return _validate_model(SUPPORTED_GENERATION_MODELS, model_name)


def _validate_allowed_origins(origins: list[str] | None) -> list[str] | None:
    if origins is None:
        return None
    for origin in origins:
        if not _ORIGIN_RE.match(origin):
            raise ValueError("Invalid origin: must be http(s)://hostname (no path or trailing slash)")
    return origins


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

    @model_validator(mode="after")
    def overlap_less_than_size(self):
        if self.chunk_size is not None and self.chunk_overlap is not None:
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return self
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1, le=100)
    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_enabled: bool | None = None
    rerank_model: str | None = None
    rerank_candidates: int | None = Field(None, ge=1, le=500)
    rerank_threshold: float | None = Field(None, ge=0.0, le=1.0)
    keyword_search_enabled: bool | None = None
    sample_questions: list[str] | None = Field(None, max_length=10)
    history_turns: int | None = Field(None, ge=0, le=50)
    max_tokens: int | None = Field(None, ge=1, le=8192)
    hosted_chat: bool | None = None
    allowed_origins: list[str] | None = Field(None, max_length=50)
    budget_limit: float | None = Field(None, ge=0.0)
    custom_anthropic_key: str | None = None
    custom_voyage_key: str | None = None

    @field_validator("sample_questions")
    @classmethod
    def validate_sample_question_length(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for q in v:
                if len(q) > 500:
                    raise ValueError("Each sample question must be 500 characters or fewer")
        return v

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
    max_tokens: int
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
