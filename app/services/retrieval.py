import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentChunk
from app.models.rag_model import RagModel
from app.services.embedder import embed_query
from app.services.reranker import rerank
from app.telemetry import tracer

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
    is_neighbor: bool = False

    @property
    def retrieval_method(self) -> str:
        """How this chunk was retrieved (reranker is orthogonal — just re-orders)."""
        if self.is_neighbor:
            return "neighbor"
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
    embed_tokens: int = 0


async def _vector_search(
    session: AsyncSession,
    model: RagModel,
    query_embedding: list[float],
    threshold_distance: float | None,
    limit: int,
) -> list[tuple[ContentChunk, float]]:
    """Retrieve chunks by cosine similarity, optionally cut at threshold_distance."""
    distance_col = ContentChunk.embedding.cosine_distance(query_embedding).label("distance")
    stmt = (
        select(ContentChunk, distance_col)
        .where(ContentChunk.model_id == model.id)
        .order_by(distance_col)
        .limit(limit)
    )
    if threshold_distance is not None:
        stmt = stmt.where(ContentChunk.embedding.cosine_distance(query_embedding) <= threshold_distance)
    with tracer.start_as_current_span(
        "retrieval.vector_search",
        attributes={"db.system": "postgresql", "retrieval.candidate_limit": limit},
    ) as span:
        result = await session.execute(stmt)
        rows = [(chunk, score) for chunk, score in result.all()]
        span.set_attribute("retrieval.results_count", len(rows))
        return rows


async def _keyword_search(
    session: AsyncSession,
    model: RagModel,
    query: str,
    limit: int,
) -> list[tuple[ContentChunk, float]]:
    """Retrieve chunks by full-text keyword search."""
    if not query.strip():
        return []
    ts_query = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank(ContentChunk.search_vector, ts_query).label("rank")
    stmt = (
        select(ContentChunk, rank)
        .where(ContentChunk.model_id == model.id)
        .where(ContentChunk.search_vector.op("@@")(ts_query))
        .order_by(rank.desc())
        .limit(limit)
    )
    with tracer.start_as_current_span(
        "retrieval.keyword_search",
        attributes={"db.system": "postgresql", "retrieval.candidate_limit": limit},
    ) as span:
        result = await session.execute(stmt)
        rows = [(chunk, score) for chunk, score in result.all()]
        span.set_attribute("retrieval.results_count", len(rows))
        return rows


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


async def _rerank_chunks(
    model: RagModel,
    query: str,
    chunks: list[ContentChunk],
) -> tuple[list[ContentChunk], dict[int, float], int]:
    """Rerank chunks using Voyage, applying score threshold. Returns (chunks, scores, tokens)."""
    if not model.reranker_enabled or len(chunks) <= 1:
        return chunks, {}, 0

    rerank_result = await rerank(
        query=query,
        documents=[c.content for c in chunks],
        model=model.rerank_model,
        top_k=model.top_k,
        voyage_api_key=model.custom_voyage_key,
    )
    rerank_scores = {chunks[i].id: s for i, s in zip(rerank_result.indices, rerank_result.scores)}
    reranked = [chunks[i] for i in rerank_result.indices]

    if model.rerank_threshold and model.rerank_threshold > 0:
        reranked = [c for c in reranked if rerank_scores.get(c.id, 0) >= model.rerank_threshold]

    return reranked, rerank_scores, rerank_result.total_tokens


async def _expand_neighbours(
    session: AsyncSession,
    model: RagModel,
    chunks: list[ContentChunk],
) -> tuple[list[ContentChunk], set[int]]:
    """Pull each hit's adjacent chunks back in, so context split across a
    boundary arrives whole. Neighbours are inserted next to the hit that pulled
    them in, preserving rank order; the hits themselves are never displaced.
    """
    radius = model.neighbor_radius or 0
    if radius <= 0 or not chunks:
        return chunks, set()

    wanted = {
        (c.source_identifier, c.position + offset)
        for c in chunks
        for offset in range(-radius, radius + 1)
        if offset
    }
    have = {(c.source_identifier, c.position) for c in chunks}
    wanted -= have
    if not wanted:
        return chunks, set()

    rows = (await session.execute(
        select(ContentChunk).where(
            ContentChunk.model_id == model.id,
            tuple_(ContentChunk.source_identifier, ContentChunk.position).in_(wanted),
        )
    )).scalars().all()
    by_key = {(r.source_identifier, r.position): r for r in rows}

    expanded: list[ContentChunk] = []
    seen: set[int] = set()
    for chunk in chunks:
        for offset in range(-radius, radius + 1):
            neighbour = (
                chunk if offset == 0
                else by_key.get((chunk.source_identifier, chunk.position + offset))
            )
            if neighbour is not None and neighbour.id not in seen:
                seen.add(neighbour.id)
                expanded.append(neighbour)
    added = {c.id for c in expanded} - {c.id for c in chunks}
    return expanded, added


async def retrieve_with_threshold(
    session: AsyncSession,
    model: RagModel,
    query: str,
    limit: int | None = None,
    context: str | None = None,
) -> RetrievalResult:
    """Hybrid retrieval: vector similarity + keyword search merged with RRF.

    When reranker is enabled, fetches a larger candidate set then reranks down to top_k.

    `context` (the previous conversation turn) is prepended to the *embedded* text
    only, so a follow-up like "well where does he work" lands near its subject.
    Keyword search and reranking still see the bare query: BM25 over a whole prior
    turn matches too broadly, and reranking on the bare question is what keeps a
    topic switch from dragging the old subject forward.
    """
    embed_text = f"{context}\n{query}" if context else query
    query_embed = await embed_query(embed_text, model=model.embedding_model, voyage_api_key=model.custom_voyage_key)

    # Cosine distance is a crude relevance proxy (r = -0.55 against rerank
    # scores on production traffic). When a reranker will judge the candidates
    # anyway, a hard distance cutoff only starves it — a vague query embeds far
    # from everything, and the answer gets discarded before the better scorer
    # ever sees it. So the cutoff applies only when no reranker runs;
    # rerank_threshold is the precision floor otherwise.
    threshold_distance = None if model.reranker_enabled else 1.0 - model.similarity_threshold

    candidate_limit = model.top_k
    if model.reranker_enabled:
        candidate_limit = model.rerank_candidates or model.top_k * RERANK_CANDIDATE_MULTIPLIER

    t0 = time.perf_counter()

    # Run searches
    vector_rows = await _vector_search(session, model, query_embed.embedding, threshold_distance, candidate_limit)

    if model.keyword_search_enabled:
        keyword_rows = await _keyword_search(session, model, query, candidate_limit)
    else:
        keyword_rows = []

    # Merge with RRF
    chunks, distances, keyword_ranks = _rrf_merge(vector_rows, keyword_rows, candidate_limit)

    logger.info("retrieval", extra={
        "vector": len(vector_rows), "keyword": len(keyword_rows),
        "merged": len(chunks), "threshold": model.similarity_threshold,
        "duration_ms": round((time.perf_counter() - t0) * 1000),
    })

    chunks, rerank_scores, rerank_tokens = await _rerank_chunks(model, query, chunks)
    # Apply the caller's cutoff to genuine hits first. Expanding then truncating
    # would let neighbours push real reranked hits out of the result.
    if limit is not None:
        chunks = chunks[:limit]
    chunks, neighbour_ids = await _expand_neighbours(session, model, chunks)

    scores = [
        ChunkScore(
            chunk_id=c.id,
            distance=round(distances[c.id], 4) if c.id in distances else 1.0,
            rerank_score=round(rerank_scores[c.id], 4) if c.id in rerank_scores else None,
            keyword_rank=keyword_ranks.get(c.id),
            is_neighbor=c.id in neighbour_ids,
        )
        for c in chunks
    ]

    return RetrievalResult(
        chunks=chunks, scores=scores,
        rerank_tokens=rerank_tokens, embed_tokens=query_embed.total_tokens,
    )
