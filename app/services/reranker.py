import hashlib
import logging
import time
from dataclasses import dataclass

import voyageai

from app.config import settings

logger = logging.getLogger("ragr.reranker")

_platform_client: voyageai.AsyncClient | None = None

# TTL cache for custom-key clients: hash(key) -> (client, created_at)
_client_cache: dict[str, tuple[voyageai.AsyncClient, float]] = {}
_CLIENT_TTL = 300  # 5 minutes


def _get_client(api_key: str | None = None) -> voyageai.AsyncClient:
    if api_key:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        entry = _client_cache.get(key_hash)
        if entry and (time.monotonic() - entry[1]) < _CLIENT_TTL:
            return entry[0]
        client = voyageai.AsyncClient(api_key=api_key, timeout=30)
        _client_cache[key_hash] = (client, time.monotonic())
        return client
    global _platform_client
    if _platform_client is None:
        _platform_client = voyageai.AsyncClient(api_key=settings.voyage_api_key, timeout=30)
    return _platform_client


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
    logger.info(
        "rerank %.0fms model=%s candidates=%d top_k=%d tokens=%d",
        (time.perf_counter() - t0) * 1000,
        model,
        len(documents),
        top_k,
        result.total_tokens,
    )
    return RerankResult(
        indices=[r.index for r in result.results],
        scores=[r.relevance_score for r in result.results],
        total_tokens=result.total_tokens,
    )
