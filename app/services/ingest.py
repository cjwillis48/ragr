import hashlib
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentChunk
from app.models.ingestion_source import IngestionSource
from app.models.rag_model import RagModel
from app.services.budget import estimate_embedding_cost
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts


@dataclass
class IngestResult:
    chunk_count: int
    skipped: bool
    embedding_cost: float


async def ingest_content(
    session: AsyncSession,
    model: RagModel,
    content: str,
    source_identifier: str,
    content_type: str = "text",
    source_url: str = "",
) -> IngestResult:
    """Ingest content for a model.

    Idempotent: if the content hash matches the existing source, skip.
    If changed, delete old chunks and re-embed.
    """
    hash_input = f"{content}:chunk_size={model.chunk_size}:chunk_overlap={model.chunk_overlap}"
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    # Check for existing ingestion source
    result = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.source_identifier == source_identifier,
        )
    )
    existing = result.scalar_one_or_none()

    if existing and existing.content_hash == content_hash:
        return IngestResult(chunk_count=existing.chunk_count, skipped=True, embedding_cost=0.0)

    # If source exists but hash changed, delete old chunks
    if existing:
        await session.execute(
            delete(ContentChunk).where(
                ContentChunk.model_id == model.id,
                ContentChunk.source_identifier == source_identifier,
            )
        )

    # Chunk the content
    chunks = chunk_text(content, model.chunk_size, model.chunk_overlap)
    if not chunks:
        return IngestResult(chunk_count=0, skipped=False, embedding_cost=0.0)

    # Embed all chunks
    embed = await embed_texts(chunks, model=model.embedding_model)

    # Store chunks
    for chunk_text_str, embedding in zip(chunks, embed.embeddings):
        chunk = ContentChunk(
            model_id=model.id,
            content=chunk_text_str,
            embedding=embedding,
            source_url=source_url,
            source_identifier=source_identifier,
            content_type=content_type,
        )
        session.add(chunk)

    # Upsert ingestion source (handles race with pending row from upload endpoint)
    embedding_cost = estimate_embedding_cost(model.embedding_model, embed.total_tokens)

    stmt = pg_insert(IngestionSource).values(
        model_id=model.id,
        source_identifier=source_identifier,
        content_hash=content_hash,
        chunk_count=len(chunks),
        source_url=source_url,
        content_type=content_type,
        status="complete",
        embedding_cost=embedding_cost,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_model_source",
        set_={
            "content_hash": stmt.excluded.content_hash,
            "chunk_count": stmt.excluded.chunk_count,
            "source_url": stmt.excluded.source_url,
            "content_type": stmt.excluded.content_type,
            "status": stmt.excluded.status,
            "embedding_cost": stmt.excluded.embedding_cost,
        },
    )
    await session.execute(stmt)
    await session.commit()

    return IngestResult(chunk_count=len(chunks), skipped=False, embedding_cost=embedding_cost)
