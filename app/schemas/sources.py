from datetime import datetime

from pydantic import BaseModel, Field


class SourceResponse(BaseModel):
    id: int
    source_identifier: str
    source_url: str
    content_type: str
    status: str
    chunk_count: int
    embedding_cost: float
    ingested_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SourceListResponse(BaseModel):
    model_slug: str
    sources: list[SourceResponse]
    total: int


class CreateSourceRequest(BaseModel):
    """Unified source creation request. Provide content, url, or urls."""
    source_identifier: str | None = None
    content: str | None = Field(default=None, min_length=1)
    url: str | None = None
    urls: list[str] | None = None
    content_type: str = "text"
    source_url: str = ""


class CreateSourceResponse(BaseModel):
    source_identifier: str
    status: str
    chunks_created: int | None = None
    skipped: bool = False
    message: str
