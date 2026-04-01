import logging
import time
from dataclasses import dataclass

import voyageai

from app.config import settings
from app.services.client_cache import ClientCache


@dataclass
class EmbedResult:
    embeddings: list[list[float]]
    total_tokens: int

logger = logging.getLogger("ragr.embedder")

_clients = ClientCache(
    platform_factory=lambda: voyageai.AsyncClient(api_key=settings.voyage_api_key, timeout=30),
    custom_factory=lambda key: voyageai.AsyncClient(api_key=key, timeout=30),
)


def _get_client(api_key: str | None = None) -> voyageai.AsyncClient:
    return _clients.get(api_key)


async def embed_texts(
    texts: list[str], model: str = "voyage-4-lite", batch_size: int = 128,
    voyage_api_key: str | None = None,
) -> EmbedResult:
    """Embed a list of texts using Voyage AI.

    Processes in batches to avoid Voyage API payload limits on large ingestion jobs.
    """
    if not texts:
        return EmbedResult(embeddings=[], total_tokens=0)
    client = _get_client(voyage_api_key)

    if len(texts) <= batch_size:
        result = await client.embed(texts, model=model, input_type="document")
        return EmbedResult(embeddings=result.embeddings, total_tokens=result.total_tokens)

    all_embeddings: list[list[float]] = []
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info("Embedding batch %d/%d (%d chunks)", i // batch_size + 1, -(-len(texts) // batch_size), len(batch))
        result = await client.embed(batch, model=model, input_type="document")
        all_embeddings.extend(result.embeddings)
        total_tokens += result.total_tokens

    return EmbedResult(embeddings=all_embeddings, total_tokens=total_tokens)


async def embed_query(text: str, model: str = "voyage-4-lite", voyage_api_key: str | None = None) -> list[float]:
    """Embed a single query text for retrieval."""
    client = _get_client(voyage_api_key)
    t0 = time.perf_counter()
    result = await client.embed([text], model=model, input_type="query")
    logger.info("embed_query %.0fms tokens=%d", (time.perf_counter() - t0) * 1000, result.total_tokens)
    return result.embeddings[0]
