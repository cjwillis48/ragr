import asyncio
import hashlib
import time
from dataclasses import dataclass
from functools import partial

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentChunk
from app.models.ingestion_source import IngestionSource
from app.models.rag_model import RagModel
from app.services.budget import estimate_embedding_cost
from app.services.chunker import chunk_code, chunk_text
from app.services.code import language_for_path
from app.services.embedder import embed_texts


@dataclass
class IngestResult:
    chunk_count: int
    skipped: bool
    embedding_cost: float
    chunk_ms: int = 0
    embed_ms: int = 0
    db_ms: int = 0


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
    def _compute_hash() -> str:
        h = f"{content}:chunk_size={model.chunk_size}:chunk_overlap={model.chunk_overlap}:embedding={model.embedding_model}"
        return hashlib.sha256(h.encode()).hexdigest()

    content_hash = _compute_hash()

    # Serialize concurrent ingests of the same source. An editor save (or a
    # watcher) can fire several PUTs for one file in quick succession; without
    # this, each request deletes old chunks early then inserts new ones after a
    # slow embed, so concurrent PUTs stack into duplicate chunk rows. This
    # transaction-scoped advisory lock makes the check-and-replace atomic per
    # (model, source) — losers re-read the now-current hash and hit the skip
    # path below, so blind re-PUTs stay idempotent.
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_model_id, hashtext(:lock_identifier))"),
        {"lock_model_id": model.id, "lock_identifier": source_identifier},
    )

    # Check for existing ingestion source
    result = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.source_identifier == source_identifier,
        )
    )
    existing = result.scalar_one_or_none()

    if existing and existing.content_hash == content_hash:
        if existing.status != "complete":
            existing.status = "complete"
            await session.commit()
        return IngestResult(chunk_count=existing.chunk_count, skipped=True, embedding_cost=0.0)

    try:
        # Skip fsync wait on commit — ingested content is re-derivable from raw_content,
        # so durability of individual chunk inserts isn't critical. Big speedup on disk-bound writes.
        await session.execute(text("SET LOCAL synchronous_commit = OFF"))

        # If source exists but hash changed, delete old chunks
        if existing:
            await session.execute(
                delete(ContentChunk).where(
                    ContentChunk.model_id == model.id,
                    ContentChunk.source_identifier == source_identifier,
                )
            )

        # Chunk the content (run in thread to avoid blocking the event loop).
        # Code uses a line-aware chunker that also returns per-chunk start/end lines.
        t_chunk = time.perf_counter()
        loop = asyncio.get_running_loop()
        if content_type == "code":
            code_chunks = await loop.run_in_executor(
                None, partial(chunk_code, content, model.chunk_size, model.chunk_overlap)
            )
            chunks = [c[0] for c in code_chunks]
            language = language_for_path(source_identifier)
            chunk_metadatas: list[dict] = [
                {
                    "file_path": source_identifier,
                    "start_line": start,
                    "end_line": end,
                    "language": language,
                }
                for _, start, end in code_chunks
            ]
        else:
            chunks = await loop.run_in_executor(
                None, partial(chunk_text, content, model.chunk_size, model.chunk_overlap)
            )
            chunk_metadatas = [{} for _ in chunks]
        chunk_ms = round((time.perf_counter() - t_chunk) * 1000)
        if not chunks:
            return IngestResult(chunk_count=0, skipped=False, embedding_cost=0.0, chunk_ms=chunk_ms)

        # Embed all chunks
        t_embed = time.perf_counter()
        embed = await embed_texts(chunks, model=model.embedding_model, voyage_api_key=model.custom_voyage_key)
        embed_ms = round((time.perf_counter() - t_embed) * 1000)

        # Store chunks (single multi-row INSERT instead of per-row session.add())
        t_db = time.perf_counter()
        chunk_rows = [
            {
                "model_id": model.id,
                "content": chunk_text_str,
                "embedding": embedding,
                "search_vector": func.to_tsvector("english", chunk_text_str),
                "source_url": source_url,
                "source_identifier": source_identifier,
                "content_type": content_type,
                "metadata_": meta,
            }
            for chunk_text_str, embedding, meta in zip(chunks, embed.embeddings, chunk_metadatas)
        ]
        await session.execute(pg_insert(ContentChunk).values(chunk_rows))

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
            raw_content=content,
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
                "raw_content": stmt.excluded.raw_content,
            },
        )
        await session.execute(stmt)
        await session.commit()
        db_ms = round((time.perf_counter() - t_db) * 1000)
    except Exception:
        await session.rollback()
        raise

    return IngestResult(
        chunk_count=len(chunks), skipped=False, embedding_cost=embedding_cost,
        chunk_ms=chunk_ms, embed_ms=embed_ms, db_ms=db_ms,
    )
