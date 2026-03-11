from datetime import datetime

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=255)


class ApiKeyCreateResponse(BaseModel):
    id: int
    label: str
    key_prefix: str
    raw_key: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ApiKeyRead(BaseModel):
    id: int
    label: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None

    model_config = {"from_attributes": True}
