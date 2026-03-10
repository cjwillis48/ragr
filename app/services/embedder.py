import logging
import time
from dataclasses import dataclass

import voyageai

from app.config import settings


@dataclass
class EmbedResult:
    embeddings: list[list[float]]
    total_tokens: int

_client: voyageai.AsyncClient | None = None

logger = logging.getLogger("ragr.embedder")

def _get_client() -> voyageai.AsyncClient:
    global _client
    if _client is None:
        _client = voyageai.AsyncClient(api_key=settings.voyage_api_key, timeout=30)
    return _client



async def embed_texts(
    texts: list[str], model: str = "voyage-4-lite", batch_size: int = 128
) -> EmbedResult:
    """Embed a list of texts using Voyage AI.

    Processes in batches to avoid Voyage API payload limits on large ingestion jobs.
    """
    if not texts:
        return EmbedResult(embeddings=[], total_tokens=0)
    client = _get_client()

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


async def embed_query(text: str, model: str = "voyage-4-lite") -> list[float]:
    """Embed a single query text for retrieval."""
    client = _get_client()
    t0 = time.perf_counter()
    result = await client.embed([text], model=model, input_type="query")
    logger.info("embed_query %.0fms tokens=%d", (time.perf_counter() - t0) * 1000, result.total_tokens)
    return result.embeddings[0]
