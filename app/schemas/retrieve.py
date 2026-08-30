from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50, description="Max chunks to return after rerank")


class ChunkScoreOut(BaseModel):
    distance: float
    rerank_score: float | None = None
    keyword_rank: int | None = None
    retrieval_method: str


class RetrievedChunk(BaseModel):
    id: int
    content: str
    source_identifier: str
    source_url: str
    content_type: str
    metadata: dict = Field(default_factory=dict)
    score: ChunkScoreOut


class RetrieveTokens(BaseModel):
    embedding: int
    rerank: int


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    tokens: RetrieveTokens
    cost: str
