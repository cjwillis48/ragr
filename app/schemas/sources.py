from datetime import datetime

from pydantic import BaseModel, Field, field_validator


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


class ChunkResponse(BaseModel):
    id: int
    content: str
    source_url: str
    content_type: str
    ingested_at: datetime

    model_config = {"from_attributes": True}


class ChunkListResponse(BaseModel):
    source_identifier: str
    chunks: list[ChunkResponse]
    total: int


def _validate_http_url(url: str) -> str:
    """Validate that a URL uses http or https scheme."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid URL '{url}': only http/https URLs are allowed")
    return url


class CreateSourceRequest(BaseModel):
    """Unified source creation request. Provide content, url, or urls."""
    source_identifier: str | None = None
    content: str | None = Field(default=None, min_length=1)
    url: str | None = None
    urls: list[str] | None = None
    content_type: str = "text"
    source_url: str = ""

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        if v is not None:
            _validate_http_url(v)
        return v

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for url in v:
                _validate_http_url(url)
        return v


class CreateSourceResponse(BaseModel):
    source_identifier: str
    status: str
    chunks_created: int | None = None
    skipped: bool = False
    message: str


class PresignedFileRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"


class PresignedUploadRequest(BaseModel):
    files: list[PresignedFileRequest]


class PresignedFileInfo(BaseModel):
    filename: str
    object_key: str
    upload_url: str
    content_type: str


class PresignedUploadResponse(BaseModel):
    upload_id: str
    files: list[PresignedFileInfo]


class ConfirmFileInfo(BaseModel):
    filename: str
    object_key: str


class ConfirmUploadRequest(BaseModel):
    upload_id: str
    files: list[ConfirmFileInfo]


class CrawlRequest(BaseModel):
    url: str
    max_pages: int = Field(50, ge=1, le=200)
    max_depth: int = Field(3, ge=1, le=5)
    prefix: str | None = None
    exclude_patterns: list[str] | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        return _validate_http_url(v)


class CrawlResponse(BaseModel):
    status: str
    message: str
    pages_queued: int
