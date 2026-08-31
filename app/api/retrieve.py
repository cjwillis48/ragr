import logging
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.dependencies import require_chat_auth
from app.models.rag_model import RagModel
from app.schemas.retrieve import (
    ChunkScoreOut,
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveTokens,
)
from app.services.budget import estimate_embedding_cost, estimate_rerank_cost
from app.services.rate_limit import RateLimiter
from app.services.retrieval import retrieve_with_threshold

router = APIRouter(tags=["retrieve"])
logger = logging.getLogger("ragr.retrieve")

_retrieve_limiter = RateLimiter(max_requests=settings.rate_limit_per_min, window_seconds=60)


def _client_ip(request: Request) -> str:
    direct = request.client.host if request.client else "unknown"
    if not (settings.trusted_proxy_ips and direct in settings.trusted_proxy_ips):
        return direct
    return (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or direct
    )


@router.post("/models/{slug}/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    request: Request,
    model: RagModel = Depends(require_chat_auth),
    session: AsyncSession = Depends(get_session),
):
    """Hybrid retrieval without generation. Returns raw chunks + metadata + scores.

    Intended for programmatic clients (e.g. MCP servers, agents) that want
    the raw retrieval material and will do their own synthesis.
    """
    rate_key = f"{model.id}:{_client_ip(request)}"
    if not _retrieve_limiter.is_allowed(rate_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")

    span = trace.get_current_span()
    span.set_attribute("retrieve.model.slug", model.slug)
    span.set_attribute("retrieve.model.id", model.id)
    span.set_attribute("retrieve.query_preview", body.query[:120])
    span.set_attribute("retrieve.top_k", body.top_k)

    t0 = time.perf_counter()
    try:
        result = await retrieve_with_threshold(session, model, body.query, limit=body.top_k)
    except httpx.TimeoutException:
        logger.error("embedding_timeout")
        raise HTTPException(status_code=503, detail="Embedding service timed out. Please try again.")
    except Exception:
        logger.exception("retrieve_failed")
        raise HTTPException(status_code=503, detail="Retrieval service unavailable. Please try again.")

    # top_k is applied inside retrieval, before neighbour expansion — slicing
    # here would discard reranked hits in favour of their neighbours.
    chunks = result.chunks
    scores = result.scores

    embed_cost = estimate_embedding_cost(model.embedding_model, result.embed_tokens)
    rerank_cost = estimate_rerank_cost(model.rerank_model, result.rerank_tokens) if result.rerank_tokens else 0.0
    total_cost = embed_cost + rerank_cost

    logger.info("retrieve", extra={
        "duration_ms": round((time.perf_counter() - t0) * 1000),
        "returned": len(chunks),
        "embed_tokens": result.embed_tokens,
        "rerank_tokens": result.rerank_tokens,
        "cost": total_cost,
    })

    return RetrieveResponse(
        chunks=[
            RetrievedChunk(
                id=c.id,
                content=c.content,
                source_identifier=c.source_identifier,
                source_url=c.source_url,
                content_type=c.content_type,
                metadata=c.metadata_ or {},
                score=ChunkScoreOut(
                    distance=s.distance,
                    rerank_score=s.rerank_score,
                    keyword_rank=s.keyword_rank,
                    retrieval_method=s.retrieval_method,
                ),
            )
            for c, s in zip(chunks, scores)
        ],
        tokens=RetrieveTokens(
            embedding=result.embed_tokens,
            rerank=result.rerank_tokens,
        ),
        cost=f"${total_cost:.6f}",
    )
