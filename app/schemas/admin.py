from datetime import datetime

from pydantic import BaseModel


class StatsResponse(BaseModel):
    model_slug: str
    total_chunks: int
    total_conversations: int
    unanswered_questions: int
    current_month_cost: float
    budget_limit: float
    budget_remaining: float
    total_sources: int


class PurgeResponse(BaseModel):
    model_slug: str
    chunks_deleted: int
    sources_deleted: int


class ConversationResponse(BaseModel):
    id: int
    session_id: str | None
    question: str
    answer: str
    status: str
    tokens_in: int
    tokens_out: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    model_slug: str
    conversations: list[ConversationResponse]
    total: int
    limit: int
    offset: int
