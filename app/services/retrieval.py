import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentChunk
from app.models.rag_model import RagModel
from app.services.embedder import embed_query
from app.services.reranker import rerank

logger = logging.getLogger("ragr.retrieval")


@dataclass
class RetrievalResult:
    chunks: list[ContentChunk] = field(default_factory=list)
    rerank_tokens: int = 0

# When reranking, fetch this many more candidates than top_k for the reranker to score
RERANK_CANDIDATE_MULTIPLIER = 4


async def retrieve_with_threshold(
    session: AsyncSession,
    model: RagModel,
    query: str,
) -> RetrievalResult:
    """Retrieve chunks above the similarity threshold using cosine distance.

    When reranker is enabled, fetches a larger candidate set from pgvector
    then reranks down to top_k.
    """
    query_embedding = await embed_query(query, model=model.embedding_model)
    threshold_distance = 1.0 - model.similarity_threshold

    candidate_limit = model.top_k
    if model.reranker_enabled:
        candidate_limit = model.top_k * RERANK_CANDIDATE_MULTIPLIER

    stmt = (
        select(ContentChunk)
        .where(ContentChunk.model_id == model.id)
        .where(ContentChunk.embedding.cosine_distance(query_embedding) <= threshold_distance)
        .order_by(ContentChunk.embedding.cosine_distance(query_embedding))
        .limit(candidate_limit)
    )
    t0 = time.perf_counter()
    result = await session.execute(stmt)
    chunks = list(result.scalars().all())
    logger.info(
        "retrieval model_id=%d chunks=%d threshold=%.2f limit=%d db=%.0fms query='%s'",
        model.id, len(chunks), model.similarity_threshold, candidate_limit,
        (time.perf_counter() - t0) * 1000, query[:60],
    )

    rerank_tokens = 0
    if model.reranker_enabled and len(chunks) > 1:
        rerank_result = await rerank(
            query=query,
            documents=[c.content for c in chunks],
            model=model.rerank_model,
            top_k=model.top_k,
        )
        chunks = [chunks[i] for i in rerank_result.indices]
        rerank_tokens = rerank_result.total_tokens

    return RetrievalResult(chunks=chunks, rerank_tokens=rerank_tokens)
