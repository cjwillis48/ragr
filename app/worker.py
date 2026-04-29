"""Background ingestion worker.

Polls the ingestion_jobs table for pending work and processes jobs
one at a time per concurrency slot. Run via: python -m app.worker

Uses SELECT FOR UPDATE SKIP LOCKED for safe multi-worker concurrency.
"""

import asyncio
import logging
import signal
from datetime import UTC, datetime

from sqlalchemy import select, update

import app.database as db
from app.config import settings
from app.database import _init_engine
from app.models.ingestion_job import IngestionJob
from app.models.ingestion_source import IngestionSource
from app.models.rag_model import RagModel
from app.services.html import strip_html
from app.logging_setup import configure_logging
from app.services.ingest import ingest_content
import app.services.wikipedia as wikipedia_module

configure_logging()
logger = logging.getLogger("ragr.worker")


# ---------------------------------------------------------------------------
# Job queue operations
# ---------------------------------------------------------------------------


async def claim_jobs(limit: int) -> list[IngestionJob]:
    """Claim pending jobs using SELECT FOR UPDATE SKIP LOCKED."""
    async with db.async_session() as session:
        result = await session.execute(
            select(IngestionJob)
            .where(
                IngestionJob.status == "pending",
                IngestionJob.attempts < IngestionJob.max_attempts,
            )
            .order_by(IngestionJob.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        jobs = result.scalars().all()
        for job in jobs:
            job.status = "running"
            job.attempts += 1
            job.claimed_at = datetime.now(UTC)
        await session.commit()
        for job in jobs:
            session.expunge(job)
        return jobs


async def mark_complete(job_id: int) -> None:
    async with db.async_session() as session:
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(status="complete", completed_at=datetime.now(UTC))
        )
        await session.commit()


async def mark_failed(job_id: int, error: str, attempts: int, max_attempts: int) -> None:
    async with db.async_session() as session:
        status = "failed" if attempts >= max_attempts else "pending"
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(status=status, error=error, completed_at=datetime.now(UTC) if status == "failed" else None)
        )
        await session.commit()


async def _mark_source_failed(model_id: int, source_identifier: str) -> None:
    """Mark an IngestionSource as failed using a fresh session."""
    try:
        async with db.async_session() as session:
            result = await session.execute(
                select(IngestionSource).where(
                    IngestionSource.model_id == model_id,
                    IngestionSource.source_identifier == source_identifier,
                )
            )
            src = result.scalar_one_or_none()
            if src:
                src.status = "failed"
                await session.commit()
    except Exception:
        logger.error("mark_source_failed_error", extra={"model_id": model_id, "source": source_identifier}, exc_info=True)


async def recover_stale_jobs() -> None:
    """Reset jobs stuck in 'running' past the configured timeout."""
    async with db.async_session() as session:
        now = datetime.now(UTC)
        # Regular jobs: 10 min timeout
        stale_cutoff = now.timestamp() - settings.worker_stale_job_timeout_minutes * 60
        await session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.status == "running",
                IngestionJob.job_type != "crawl",
                IngestionJob.claimed_at < datetime.fromtimestamp(stale_cutoff, tz=UTC),
            )
            .values(status="pending", claimed_at=None)
        )
        # Crawl jobs: 30 min timeout
        crawl_cutoff = now.timestamp() - settings.worker_stale_crawl_timeout_minutes * 60
        await session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.status == "running",
                IngestionJob.job_type == "crawl",
                IngestionJob.claimed_at < datetime.fromtimestamp(crawl_cutoff, tz=UTC),
            )
            .values(status="pending", claimed_at=None)
        )
        await session.commit()


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------


async def handle_url_job(job: IngestionJob) -> None:
    """Fetch a URL, strip HTML, and ingest."""
    from app.services.crawler import _fetch_page

    url = job.job_params["url"]
    source_identifier = job.job_params["source_identifier"]
    model_id = job.model_id

    async with db.async_session() as session:
        result = await session.execute(select(RagModel).where(RagModel.id == model_id))
        model = result.scalar_one()

        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        resp = await _fetch_page(url, timeout=30)
        resp.raise_for_status()

        content_length = int(resp.headers.get("content-length", 0))
        if content_length > max_bytes:
            raise ValueError(f"Response too large: {content_length} bytes")
        if len(resp.content) > max_bytes:
            raise ValueError(f"Response body too large: {len(resp.content)} bytes")

        content_type_header = resp.headers.get("content-type", "")
        if "html" in content_type_header:
            text = await asyncio.to_thread(strip_html, resp.text)
            ct = "html"
        else:
            text = resp.text
            ct = "text"

        ingest_result = await ingest_content(
            session=session, model=model, content=text,
            source_identifier=source_identifier, content_type=ct, source_url=url,
        )
        logger.info("url_ingested", extra={
            "model_id": model_id, "url": url,
            "chunks": ingest_result.chunk_count, "cost": ingest_result.embedding_cost,
            "chunk_ms": ingest_result.chunk_ms, "embed_ms": ingest_result.embed_ms, "db_ms": ingest_result.db_ms,
        })


async def handle_file_job(job: IngestionJob) -> None:
    """Ingest pre-extracted file text from IngestionSource.raw_content."""
    source_identifier = job.job_params["source_identifier"]
    content_type = job.job_params["content_type"]
    model_id = job.model_id

    async with db.async_session() as session:
        result = await session.execute(select(RagModel).where(RagModel.id == model_id))
        model = result.scalar_one()

        # Read pre-extracted text from the source record
        src_result = await session.execute(
            select(IngestionSource).where(
                IngestionSource.model_id == model_id,
                IngestionSource.source_identifier == source_identifier,
            )
        )
        src = src_result.scalar_one_or_none()
        if not src or not src.raw_content:
            raise ValueError(f"No raw_content found for source {source_identifier}")

        ingest_result = await ingest_content(
            session=session, model=model, content=src.raw_content,
            source_identifier=source_identifier, content_type=content_type,
            source_url=source_identifier,
        )
        if ingest_result.skipped:
            logger.info("file_skipped", extra={"model_id": model_id, "source": source_identifier})
        else:
            logger.info("file_ingested", extra={
                "model_id": model_id, "source": source_identifier,
                "chunks": ingest_result.chunk_count, "cost": ingest_result.embedding_cost,
                "chunk_ms": ingest_result.chunk_ms, "embed_ms": ingest_result.embed_ms, "db_ms": ingest_result.db_ms,
            })


async def handle_r2_file_job(job: IngestionJob) -> None:
    """Download from R2, extract text, ingest, delete from R2."""
    from pathlib import Path

    import pymupdf  # noqa: F401
    from app.services.r2 import delete_object, download_object

    object_key = job.job_params["object_key"]
    filename = job.job_params["filename"]
    model_id = job.model_id

    ALLOWED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf", ".csv", ".json"}

    try:
        raw = await download_object(object_key)

        # Inline text extraction (same logic as _extract_text in sources.py)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if ext == ".pdf":
            doc = pymupdf.Document(stream=raw, filetype="pdf")
            pages = [page.get_text() for page in doc]
            text = "\n\n".join(pages)
            content_type = "pdf"
        else:
            text = raw.decode("utf-8")
            if ext in (".html", ".htm"):
                text = await asyncio.to_thread(strip_html, text)
                content_type = "html"
            elif ext == ".md":
                content_type = "markdown"
            else:
                content_type = "text"

        async with db.async_session() as session:
            result = await session.execute(select(RagModel).where(RagModel.id == model_id))
            model = result.scalar_one()

            await ingest_content(
                session=session, model=model, content=text,
                source_identifier=filename, content_type=content_type, source_url=filename,
            )
        logger.info("r2_file_ingested", extra={"model_id": model_id, "filename": filename})
    finally:
        try:
            await delete_object(object_key)
        except Exception:
            logger.warning("r2_delete_failed", extra={"object_key": object_key})


async def handle_crawl_job(job: IngestionJob) -> None:
    """Run the crawler and create child URL jobs for discovered pages."""
    from app.services.crawler import CrawledPage, FailedPage, crawl_site

    model_id = job.model_id
    params = job.job_params
    page_count = 0

    async for item in crawl_site(
        root_url=params["url"],
        max_pages=params.get("max_pages", 50),
        max_depth=params.get("max_depth", 3),
        prefix=params.get("prefix"),
        exclude_patterns=params.get("exclude_patterns"),
    ):
        if isinstance(item, FailedPage):
            async with db.async_session() as session:
                existing = await session.execute(
                    select(IngestionSource).where(
                        IngestionSource.model_id == model_id,
                        IngestionSource.source_identifier == item.url,
                    )
                )
                src = existing.scalar_one_or_none()
                if src:
                    src.status = "failed"
                else:
                    session.add(IngestionSource(
                        model_id=model_id, source_identifier=item.url,
                        content_hash="", chunk_count=0, source_url=item.url,
                        content_type="html", status="failed",
                    ))
                await session.commit()
            continue

        # CrawledPage — create pending source + child URL job
        page_count += 1
        async with db.async_session() as session:
            existing = await session.execute(
                select(IngestionSource).where(
                    IngestionSource.model_id == model_id,
                    IngestionSource.source_identifier == item.url,
                )
            )
            src = existing.scalar_one_or_none()
            if src:
                src.status = "pending"
                src.raw_content = item.text
            else:
                session.add(IngestionSource(
                    model_id=model_id, source_identifier=item.url,
                    content_hash="", chunk_count=0, source_url=item.url,
                    content_type=item.content_type, status="pending",
                    raw_content=item.text,
                ))

            # Create a child job for ingesting this page
            session.add(IngestionJob(
                model_id=model_id,
                job_type="file",
                job_params={"source_identifier": item.url, "content_type": item.content_type, "parent_job_id": job.id},
            ))
            await session.commit()

    # Flip root URL from "crawling" to final status
    async with db.async_session() as session:
        root_result = await session.execute(
            select(IngestionSource).where(
                IngestionSource.model_id == model_id,
                IngestionSource.source_identifier == params["url"],
                IngestionSource.status == "crawling",
            )
        )
        root_src = root_result.scalar_one_or_none()
        if root_src:
            root_src.status = "complete" if page_count > 0 else "failed"
            await session.commit()


# ---------------------------------------------------------------------------
# Job dispatcher
# ---------------------------------------------------------------------------

_HANDLERS = {
    "url": handle_url_job,
    "file": handle_file_job,
    "r2_file": handle_r2_file_job,
    "crawl": handle_crawl_job,
}


async def process_job(job: IngestionJob) -> None:
    """Dispatch and process a single job."""
    handler = _HANDLERS.get(job.job_type)
    if not handler:
        logger.error("unknown_job_type", extra={"job_id": job.id, "job_type": job.job_type})
        await mark_failed(job.id, f"Unknown job type: {job.job_type}", job.max_attempts, job.max_attempts)
        return

    parent_job_id = job.job_params.get("parent_job_id")
    job_extra = {"job_id": job.id, "job_type": job.job_type, "model_id": job.model_id}
    if parent_job_id:
        job_extra["parent_job_id"] = parent_job_id

    try:
        await handler(job)
        await mark_complete(job.id)
        logger.info("job_complete", extra=job_extra)
    except Exception as e:
        logger.exception("job_failed", extra=job_extra)
        await mark_failed(job.id, str(e), job.attempts, job.max_attempts)
        # Mark the source as failed if all retries are exhausted
        if job.attempts >= job.max_attempts:
            source_id = job.job_params.get("source_identifier") or job.job_params.get("filename") or job.job_params.get("url")
            if source_id:
                await _mark_source_failed(job.model_id, source_id)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def main() -> None:
    _init_engine()
    logger.info("worker_starting", extra={"concurrency": settings.worker_concurrency})

    shutdown_event = asyncio.Event()
    active_tasks: set[asyncio.Task] = set()
    last_stale_check = 0.0

    def _shutdown():
        logger.info("worker_shutdown_requested")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    while not shutdown_event.is_set():
        # Periodic stale job recovery
        now = loop.time()
        if now - last_stale_check > 60:
            try:
                await recover_stale_jobs()
            except Exception:
                logger.exception("stale_job_recovery_failed")
            last_stale_check = now

        # Claim new jobs if we have capacity
        available = settings.worker_concurrency - len(active_tasks)
        if available > 0:
            try:
                jobs = await claim_jobs(limit=available)
                for job in jobs:
                    task = asyncio.create_task(process_job(job), name=f"job-{job.id}")
                    active_tasks.add(task)
            except Exception:
                logger.exception("claim_jobs_failed")

        # Wait for task completion or poll interval
        if active_tasks:
            done, _ = await asyncio.wait(
                active_tasks, timeout=settings.worker_poll_interval, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                active_tasks.discard(task)
                if task.exception():
                    logger.error("task_exception", extra={"task": task.get_name()}, exc_info=task.exception())
        else:
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=settings.worker_poll_interval)
            except asyncio.TimeoutError:
                pass

    # Graceful shutdown: wait for in-flight tasks
    if active_tasks:
        logger.info("worker_draining", extra={"tasks": len(active_tasks)})
        await asyncio.wait(active_tasks, timeout=300)

    # Close shared HTTP clients
    if wikipedia_module._wikipedia_client and not wikipedia_module._wikipedia_client.is_closed:
        await wikipedia_module._wikipedia_client.aclose()

    logger.info("worker_stopped")


if __name__ == "__main__":
    asyncio.run(main())
