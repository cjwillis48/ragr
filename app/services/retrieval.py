import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentChunk
from app.models.rag_model import RagModel
from app.services.embedder import embed_query
from app.services.reranker import rerank

logger = logging.getLogger("ragr.retrieval")

# Reciprocal Rank Fusion constant — standard value from the RRF paper
RRF_K = 60

# When reranking, fetch this many more candidates than top_k for the reranker to score
RERANK_CANDIDATE_MULTIPLIER = 4


@dataclass
class ChunkScore:
    chunk_id: int
    distance: float
    rerank_score: float | None = None
    keyword_rank: int | None = None

    @property
    def retrieval_method(self) -> str:
        """How this chunk was retrieved (reranker is orthogonal — just re-orders)."""
        has_vector = self.distance < 1.0
        has_keyword = self.keyword_rank is not None
        if has_vector and has_keyword:
            return "hybrid"
        if has_keyword:
            return "keyword"
        return "vector"


@dataclass
class RetrievalResult:
    chunks: list[ContentChunk] = field(default_factory=list)
    scores: list[ChunkScore] = field(default_factory=list)
    rerank_tokens: int = 0


async def _vector_search(
    session: AsyncSession,
    model: RagModel,
    query_embedding: list[float],
    threshold_distance: float,
    limit: int,
) -> list[tuple[ContentChunk, float]]:
    """Retrieve chunks by cosine similarity."""
    distance_col = ContentChunk.embedding.cosine_distance(query_embedding).label("distance")
    stmt = (
        select(ContentChunk, distance_col)
        .where(ContentChunk.model_id == model.id)
        .where(ContentChunk.embedding.cosine_distance(query_embedding) <= threshold_distance)
        .order_by(distance_col)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.all())


import re

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


async def _keyword_search(
    session: AsyncSession,
    model: RagModel,
    query: str,
    limit: int,
) -> list[tuple[ContentChunk, float]]:
    """Retrieve chunks by full-text keyword search using OR matching."""
    # Extract alphanumeric words, skip short ones, join with OR
    words = [w for w in _WORD_RE.findall(query.lower()) if len(w) > 1]
    if not words:
        return []
    or_expr = " | ".join(words)
    ts_query = func.to_tsquery("english", or_expr)
    rank = func.ts_rank(ContentChunk.search_vector, ts_query).label("rank")
    stmt = (
        select(ContentChunk, rank)
        .where(ContentChunk.model_id == model.id)
        .where(ContentChunk.search_vector.op("@@")(ts_query))
        .order_by(rank.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.all())


def _rrf_merge(
    vector_results: list[tuple[ContentChunk, float]],
    keyword_results: list[tuple[ContentChunk, float]],
    limit: int,
) -> tuple[list[ContentChunk], dict[int, float], dict[int, int]]:
    """Merge vector and keyword results using Reciprocal Rank Fusion.

    Returns (chunks, distances, keyword_ranks).
    """
    scores: dict[int, float] = {}
    chunk_map: dict[int, ContentChunk] = {}
    distances: dict[int, float] = {}
    keyword_ranks: dict[int, int] = {}

    for rank, (chunk, distance) in enumerate(vector_results, 1):
        chunk_map[chunk.id] = chunk
        distances[chunk.id] = distance
        scores[chunk.id] = scores.get(chunk.id, 0) + 1.0 / (RRF_K + rank)

    for rank, (chunk, _ts_rank) in enumerate(keyword_results, 1):
        chunk_map[chunk.id] = chunk
        keyword_ranks[chunk.id] = rank
        if chunk.id not in distances:
            distances[chunk.id] = 1.0  # keyword-only result, no vector distance
        scores[chunk.id] = scores.get(chunk.id, 0) + 1.0 / (RRF_K + rank)

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]
    chunks = [chunk_map[cid] for cid in sorted_ids]
    return chunks, distances, keyword_ranks


async def retrieve_with_threshold(
    session: AsyncSession,
    model: RagModel,
    query: str,
) -> RetrievalResult:
    """Hybrid retrieval: vector similarity + keyword search merged with RRF.

    When reranker is enabled, fetches a larger candidate set then reranks down to top_k.
    """
    query_embedding = await embed_query(query, model=model.embedding_model, voyage_api_key=model.custom_voyage_key)
    threshold_distance = 1.0 - model.similarity_threshold

    candidate_limit = model.top_k
    if model.reranker_enabled:
        candidate_limit = model.top_k * RERANK_CANDIDATE_MULTIPLIER

    t0 = time.perf_counter()

    # Run both searches
    vector_rows = await _vector_search(session, model, query_embedding, threshold_distance, candidate_limit)
    keyword_rows = await _keyword_search(session, model, query, candidate_limit)

    # Merge with RRF
    chunks, distances, keyword_ranks = _rrf_merge(vector_rows, keyword_rows, candidate_limit)

    logger.info(
        "retrieval model_id=%d vector=%d keyword=%d merged=%d threshold=%.2f db=%.0fms query='%s'",
        model.id, len(vector_rows), len(keyword_rows), len(chunks),
        model.similarity_threshold, (time.perf_counter() - t0) * 1000, query[:60],
    )

    rerank_tokens = 0
    rerank_scores: dict[int, float] = {}
    if model.reranker_enabled and len(chunks) > 1:
        rerank_result = await rerank(
            query=query,
            documents=[c.content for c in chunks],
            model=model.rerank_model,
            top_k=model.top_k,
            voyage_api_key=model.custom_voyage_key,
        )
        rerank_scores = {chunks[i].id: s for i, s in zip(rerank_result.indices, rerank_result.scores)}
        chunks = [chunks[i] for i in rerank_result.indices]
        rerank_tokens = rerank_result.total_tokens

    scores = [
        ChunkScore(
            chunk_id=c.id,
            distance=round(distances[c.id], 4),
            rerank_score=round(rerank_scores[c.id], 4) if c.id in rerank_scores else None,
            keyword_rank=keyword_ranks.get(c.id),
        )
        for c in chunks
    ]

    return RetrievalResult(chunks=chunks, scores=scores, rerank_tokens=rerank_tokens)
