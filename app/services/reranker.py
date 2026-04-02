import logging
import time
from dataclasses import dataclass

import voyageai

from app.config import settings
from app.services.client_cache import ClientCache

logger = logging.getLogger("ragr.reranker")

_clients = ClientCache(
    platform_factory=lambda: voyageai.AsyncClient(api_key=settings.voyage_api_key, timeout=30),
    custom_factory=lambda key: voyageai.AsyncClient(api_key=key, timeout=30),
)


def _get_client(api_key: str | None = None) -> voyageai.AsyncClient:
    return _clients.get(api_key)


@dataclass
class RerankResult:
    indices: list[int]
    scores: list[float]
    total_tokens: int


async def rerank(
    query: str,
    documents: list[str],
    model: str = "rerank-2.5-lite",
    top_k: int = 5,
    voyage_api_key: str | None = None,
) -> RerankResult:
    """Rerank documents by relevance to the query using Voyage AI."""
    client = _get_client(voyage_api_key)
    t0 = time.perf_counter()
    result = await client.rerank(query, documents, model=model, top_k=top_k)
    logger.info("rerank", extra={
        "duration_ms": round((time.perf_counter() - t0) * 1000),
        "model": model, "candidates": len(documents), "top_k": top_k, "tokens": result.total_tokens,
    })
    return RerankResult(
        indices=[r.index for r in result.results],  # type: ignore[misc] # NamedTuple .index shadows tuple.index()
        scores=[r.relevance_score for r in result.results],
        total_tokens=result.total_tokens,
    )
