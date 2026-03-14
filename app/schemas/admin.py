from datetime import datetime

from pydantic import BaseModel


class StatsResponse(BaseModel):
    model_slug: str
    total_chunks: int
    total_conversations: int
    total_messages: int
    unanswered_questions: int
    current_month_cost: float
    budget_limit: float
    budget_remaining: float
    total_sources: int


class PurgeResponse(BaseModel):
    model_slug: str
    chunks_deleted: int
    sources_deleted: int


class MessageResponse(BaseModel):
    id: int
    question: str
    answer: str
    status: str
    tokens_in: int
    tokens_out: int
    retrieved_chunks: list[dict] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationSummaryResponse(BaseModel):
    id: int
    session_id: str
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(BaseModel):
    id: int
    session_id: str
    title: str | None
    messages: list[MessageResponse]
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChunkResponse(BaseModel):
    id: int
    content: str
    source_url: str
    source_identifier: str
    content_type: str

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    model_slug: str
    conversations: list[ConversationSummaryResponse]
    total: int
    limit: int
    offset: int


class SystemPromptHistoryResponse(BaseModel):
    id: int
    prompt_text: str
    source: str
    input_text: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
