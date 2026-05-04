from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    stream: bool = False
    session_id: str | None = Field(None, max_length=64)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    response: str
    status: str  # "answered" | "unanswered" | "off_topic"
    session_id: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: str | None = None
