import asyncio
import logging

import httpx
import pymupdf  # noqa: F401 — eager import to avoid cold-start delay

# Limit concurrent background ingestion tasks to avoid exhausting the DB
# connection pool (default QueuePool size=5, overflow=10).
_ingest_semaphore = asyncio.Semaphore(3)
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, get_session
from app.dependencies import require_model_auth
from app.models.content import ContentChunk
from app.models.ingestion_source import IngestionSource
from app.models.rag_model import RagModel
from app.schemas.admin import PurgeResponse
from app.schemas.sources import (
    ChunkListResponse,
    ChunkResponse,
    ConfirmUploadRequest,
    CrawlRequest,
    CrawlResponse,
    CreateSourceRequest,
    CreateSourceResponse,
    PresignedUploadRequest,
    PresignedUploadResponse,
    PresignedFileInfo,
    SourceListResponse,
    SourceResponse,
)
from app.services.ingest import ingest_content
from app.services.r2 import is_configured as r2_is_configured

router = APIRouter(tags=["sources"])
logger = logging.getLogger("ragr.sources")


@router.get(
    "/models/{slug}/sources",
    response_model=SourceListResponse,
)
async def list_sources(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """List all ingested sources for a model."""
    result = await session.execute(
        select(IngestionSource)
        .where(IngestionSource.model_id == model.id)
        .order_by(IngestionSource.ingested_at.desc())
    )
    sources = result.scalars().all()
    return SourceListResponse(
        model_slug=model.slug,
        sources=[SourceResponse.model_validate(s) for s in sources],
        total=len(sources),
    )


@router.get(
    "/models/{slug}/sources/{source_id}",
    response_model=SourceResponse,
)
async def get_source(
    source_id: int,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Get a single source by ID."""
    result = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.id == source_id,
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return SourceResponse.model_validate(source)


@router.get(
    "/models/{slug}/sources/{source_id}/chunks",
    response_model=ChunkListResponse,
)
async def list_source_chunks(
    source_id: int,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """List all content chunks for a source. Useful for debugging ingestion."""
    result = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.id == source_id,
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    chunks_result = await session.execute(
        select(ContentChunk)
        .where(
            ContentChunk.model_id == model.id,
            ContentChunk.source_identifier == source.source_identifier,
        )
        .order_by(ContentChunk.id)
    )
    chunks = chunks_result.scalars().all()
    return ChunkListResponse(
        source_identifier=source.source_identifier,
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
        total=len(chunks),
    )


@router.delete(
    "/models/{slug}/sources/{source_id}",
    status_code=204,
)
async def delete_source(
    source_id: int,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Delete a single source and its chunks."""
    result = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.id == source_id,
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    await session.execute(
        delete(ContentChunk).where(
            ContentChunk.model_id == model.id,
            ContentChunk.source_identifier == source.source_identifier,
        )
    )
    await session.delete(source)
    await session.commit()


@router.delete(
    "/models/{slug}/sources",
    response_model=PurgeResponse,
)
async def purge_sources(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Delete all ingested content for a model (chunks + sources). Model config is preserved."""
    chunk_result = await session.execute(
        delete(ContentChunk).where(ContentChunk.model_id == model.id)
    )
    source_result = await session.execute(
        delete(IngestionSource).where(IngestionSource.model_id == model.id)
    )
    await session.commit()

    return PurgeResponse(
        model_slug=model.slug,
        chunks_deleted=chunk_result.rowcount,
        sources_deleted=source_result.rowcount,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Unified POST /sources
# ---------------------------------------------------------------------------

async def _ingest_url_background(model_id: int, url: str, source_identifier: str) -> None:
    """Fetch a URL server-side, strip HTML, and ingest."""
    async with async_session() as session:
        result = await session.execute(select(RagModel).where(RagModel.id == model_id))
        model = result.scalar_one()

        # Mark as pending first
        src_result = await session.execute(
            select(IngestionSource).where(
                IngestionSource.model_id == model_id,
                IngestionSource.source_identifier == source_identifier,
            )
        )
        src = src_result.scalar_one_or_none()
        if src:
            src.status = "pending"
            await session.commit()

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content_type_header = resp.headers.get("content-type", "")
                raw_text = resp.text

            if "html" in content_type_header:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw_text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "head"]):
                    tag.decompose()
                text = soup.get_text(separator="\n\n").strip()
                ct = "html"
            else:
                text = raw_text
                ct = "text"

            ingest_result = await ingest_content(
                session=session,
                model=model,
                content=text,
                source_identifier=source_identifier,
                content_type=ct,
                source_url=url,
            )
            logger.info(
                "URL %s: ingested %d chunks (cost: $%.6f)",
                url, ingest_result.chunk_count, ingest_result.embedding_cost,
            )
        except Exception:
            logger.exception("URL ingestion failed for %s", url)
            # Mark failed
            async with async_session() as err_session:
                err_result = await err_session.execute(
                    select(IngestionSource).where(
                        IngestionSource.model_id == model_id,
                        IngestionSource.source_identifier == source_identifier,
                    )
                )
                err_src = err_result.scalar_one_or_none()
                if err_src:
                    err_src.status = "failed"
                    await err_session.commit()


async def _ingest_file_background_with_status(
    model_id: int,
    text: str,
    source_identifier: str,
    content_type: str,
) -> None:
    """Run file ingestion in the background, updating status on completion."""
    async with async_session() as session:
        from app.models.rag_model import RagModel

        result = await session.execute(select(RagModel).where(RagModel.id == model_id))
        model = result.scalar_one()

        try:
            ingest_result = await ingest_content(
                session=session,
                model=model,
                content=text,
                source_identifier=source_identifier,
                content_type=content_type,
                source_url=source_identifier,
            )
            if ingest_result.skipped:
                logger.info("File %s: content unchanged, skipped", source_identifier)
            else:
                logger.info(
                    "File %s: ingested %d chunks (cost: $%.6f)",
                    source_identifier, ingest_result.chunk_count, ingest_result.embedding_cost,
                )
        except Exception:
            logger.exception("File ingestion failed for %s", source_identifier)
            async with async_session() as err_session:
                err_result = await err_session.execute(
                    select(IngestionSource).where(
                        IngestionSource.model_id == model_id,
                        IngestionSource.source_identifier == source_identifier,
                    )
                )
                err_src = err_result.scalar_one_or_none()
                if err_src:
                    err_src.status = "failed"
                    await err_session.commit()


@router.post(
    "/models/{slug}/sources",
    response_model=list[CreateSourceResponse],
    status_code=200,
)
async def create_source(
    body: CreateSourceRequest,
    response: Response,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Unified text/URL ingest. Use POST /sources/upload for file upload.

    - `content` present → synchronous text ingest (200), requires `source_identifier`
    - `url` present → async URL fetch + ingest (202), `source_identifier` derived from URL if omitted
    - `urls` present → async batch URL fetch + ingest (202), `source_identifier` derived from each URL
    """
    has_content = body.content is not None
    has_url = body.url is not None
    has_urls = body.urls is not None and len(body.urls) > 0
    provided = sum([has_content, has_url, has_urls])

    if provided == 0:
        raise HTTPException(status_code=422, detail="Provide one of 'content', 'url', or 'urls'")
    if provided > 1:
        raise HTTPException(status_code=422, detail="Provide only one of 'content', 'url', or 'urls'")

    if has_content:
        if not body.source_identifier:
            raise HTTPException(status_code=422, detail="'source_identifier' is required for text ingest")
        result = await ingest_content(
            session=session,
            model=model,
            content=body.content,
            source_identifier=body.source_identifier,
            content_type=body.content_type,
            source_url=body.source_url,
        )
        response.status_code = 200
        return [CreateSourceResponse(
            source_identifier=body.source_identifier,
            status="complete",
            chunks_created=result.chunk_count,
            skipped=result.skipped,
            message="Content unchanged, skipped re-ingestion" if result.skipped else f"Ingested {result.chunk_count} chunks",
        )]

    # Normalise single url into a list; source_identifier only applies to single url
    urls = body.urls if has_urls else [body.url]
    source_ids = [url for url in urls] if has_urls else [body.source_identifier or body.url]

    # Single query to find all existing sources
    existing_result = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.source_identifier.in_(source_ids),
        )
    )
    existing_map = {s.source_identifier: s for s in existing_result.scalars().all()}

    results = []
    task_args = []

    for url, source_id in zip(urls, source_ids):
        existing = existing_map.get(source_id)
        if existing:
            existing.status = "pending"
        else:
            session.add(IngestionSource(
                model_id=model.id,
                source_identifier=source_id,
                content_hash="",
                chunk_count=0,
                source_url=url,
                content_type="html",
                status="pending",
            ))

        task_args.append((url, source_id))
        results.append(CreateSourceResponse(
            source_identifier=source_id,
            status="pending",
            message=f"URL ingestion started for {url}",
        ))

    await session.commit()

    # Fire background tasks after commit so they don't contend with the response
    for url, source_id in task_args:
        asyncio.create_task(
            _ingest_url_background(
                model_id=model.id,
                url=url,
                source_identifier=source_id,
            )
        )

    response.status_code = 202
    return results


def _extract_text(filename: str, raw: bytes) -> tuple[str, str]:
    """Extract text and content_type from raw file bytes. Raises HTTPException on failure."""
    if filename.lower().endswith(".pdf"):
        import pymupdf
        try:
            doc = pymupdf.Document(stream=raw, filetype="pdf")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not parse PDF")
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(pages)
        page_count = len(pages)
        char_count = len(text.strip())
        logger.info("PDF %s: %d pages, %d chars extracted", filename, page_count, char_count)
        if char_count < 100:
            raise HTTPException(
                status_code=422,
                detail=f"PDF appears to be scanned/image-based",
            )
        return text, "pdf"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    if filename.lower().endswith((".html", ".htm")):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator="\n\n"), "html"
    elif filename.lower().endswith(".md"):
        return text, "markdown"
    return text, "text"


@router.post(
    "/models/{slug}/sources/upload",
    response_model=list[CreateSourceResponse],
    status_code=202,
)
async def upload_source(
    files: list[UploadFile],
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Upload one or more files to ingest. Supports .txt, .md, .html, .pdf files. Returns 202."""
    import time

    prepared_files: list[tuple[str, str, str]] = []
    results = []

    t0 = time.monotonic()
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required for all files")

        raw = await file.read()
        t_read = time.monotonic()
        text, content_type = _extract_text(file.filename, raw)
        t_extract = time.monotonic()
        logger.info(
            "upload %s: read=%.1fms extract=%.1fms size=%dKB",
            file.filename,
            (t_read - t0) * 1000,
            (t_extract - t_read) * 1000,
            len(raw) // 1024,
        )
        prepared_files.append((file.filename, text, content_type))
        t0 = time.monotonic()

    t_db = time.monotonic()
    for filename, text, content_type in prepared_files:
        src_result = await session.execute(
            select(IngestionSource).where(
                IngestionSource.model_id == model.id,
                IngestionSource.source_identifier == filename,
            )
        )
        existing = src_result.scalar_one_or_none()
        if existing:
            existing.status = "pending"
        else:
            pending_source = IngestionSource(
                model_id=model.id,
                source_identifier=filename,
                content_hash="",
                chunk_count=0,
                source_url=filename,
                content_type=content_type,
                status="pending",
            )
            session.add(pending_source)

        results.append(CreateSourceResponse(
            source_identifier=filename,
            status="pending",
            message=f"Ingestion started for {filename}",
        ))

    await session.commit()
    logger.info("upload db upserts: %.1fms for %d files", (time.monotonic() - t_db) * 1000, len(prepared_files))

    for filename, text, content_type in prepared_files:
        asyncio.create_task(
            _ingest_file_background_with_status(
                model_id=model.id,
                text=text,
                source_identifier=filename,
                content_type=content_type,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Presigned R2 upload flow
# ---------------------------------------------------------------------------


@router.post(
    "/models/{slug}/sources/upload/presign",
    response_model=PresignedUploadResponse,
)
async def presign_upload(
    body: PresignedUploadRequest,
    model: RagModel = Depends(require_model_auth),
):
    """Generate presigned R2 PUT URLs for each file. Browser uploads directly to R2."""
    if not r2_is_configured():
        raise HTTPException(status_code=501, detail="R2 storage is not configured")

    import uuid
    from app.services.r2 import generate_presigned_upload_url

    upload_id = str(uuid.uuid4())
    files = []
    for f in body.files:
        object_key = f"uploads/{model.id}/{upload_id}/{f.filename}"
        url = generate_presigned_upload_url(object_key, f.content_type)
        files.append(PresignedFileInfo(
            filename=f.filename,
            object_key=object_key,
            upload_url=url,
            content_type=f.content_type,
        ))

    return PresignedUploadResponse(upload_id=upload_id, files=files)


@router.post(
    "/models/{slug}/sources/upload/confirm",
    response_model=list[CreateSourceResponse],
    status_code=202,
)
async def confirm_upload(
    body: ConfirmUploadRequest,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Confirm files have been uploaded to R2. Triggers background ingestion."""
    if not r2_is_configured():
        raise HTTPException(status_code=501, detail="R2 storage is not configured")

    results = []
    for f in body.files:
        # Validate object key belongs to this model
        if not f.object_key.startswith(f"uploads/{model.id}/"):
            raise HTTPException(status_code=403, detail=f"Object key does not belong to this model: {f.object_key}")

        src_result = await session.execute(
            select(IngestionSource).where(
                IngestionSource.model_id == model.id,
                IngestionSource.source_identifier == f.filename,
            )
        )
        existing = src_result.scalar_one_or_none()
        if existing:
            existing.status = "pending"
        else:
            session.add(IngestionSource(
                model_id=model.id,
                source_identifier=f.filename,
                content_hash="",
                chunk_count=0,
                source_url=f.filename,
                content_type="pending",
                status="pending",
            ))

        results.append(CreateSourceResponse(
            source_identifier=f.filename,
            status="pending",
            message=f"Ingestion started for {f.filename}",
        ))

    await session.commit()

    for f in body.files:
        asyncio.create_task(
            _ingest_r2_file_background(
                model_id=model.id,
                object_key=f.object_key,
                filename=f.filename,
            )
        )

    return results


async def _ingest_r2_file_background(
    model_id: int,
    object_key: str,
    filename: str,
) -> None:
    """Download file from R2, extract text, ingest, then delete from R2."""
    from app.services.r2 import download_object, delete_object

    async with _ingest_semaphore:
        try:
            # Download and extract text WITHOUT holding a DB connection.
            raw = await asyncio.get_event_loop().run_in_executor(
                None, download_object, object_key
            )
            text, content_type = _extract_text(filename, raw)

            # Now open a session only for the DB-bound ingestion work.
            async with async_session() as session:
                result = await session.execute(select(RagModel).where(RagModel.id == model_id))
                model = result.scalar_one()

                await ingest_content(
                    session=session,
                    model=model,
                    content=text,
                    source_identifier=filename,
                    content_type=content_type,
                    source_url=filename,
                )
            logger.info("R2 file %s: ingested successfully", filename)
        except Exception:
            logger.exception("R2 file ingestion failed for %s", filename)
            async with async_session() as err_session:
                err_result = await err_session.execute(
                    select(IngestionSource).where(
                        IngestionSource.model_id == model_id,
                        IngestionSource.source_identifier == filename,
                    )
                )
                err_src = err_result.scalar_one_or_none()
                if err_src:
                    err_src.status = "failed"
                    await err_session.commit()
        finally:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, delete_object, object_key
                )
            except Exception:
                logger.warning("Failed to delete R2 object %s", object_key)


# ---------------------------------------------------------------------------
# Site crawl
# ---------------------------------------------------------------------------


async def _crawl_site_background(model_id: int, crawl_request: CrawlRequest) -> None:
    """Crawl a site and ingest all discovered pages."""
    from app.services.crawler import crawl_site

    try:
        pages = await crawl_site(
            root_url=crawl_request.url,
            max_pages=crawl_request.max_pages,
            max_depth=crawl_request.max_depth,
            prefix=crawl_request.prefix,
            exclude_patterns=crawl_request.exclude_patterns,
        )
    except Exception:
        logger.exception("Crawl failed for %s", crawl_request.url)
        return

    for page in pages:
        async with _ingest_semaphore:
            async with async_session() as session:
                result = await session.execute(select(RagModel).where(RagModel.id == model_id))
                model = result.scalar_one()

                try:
                    await ingest_content(
                        session=session,
                        model=model,
                        content=page.text,
                        source_identifier=page.url,
                        content_type=page.content_type,
                        source_url=page.url,
                    )
                    logger.info("Crawl ingest complete: %s", page.url)
                except Exception:
                    logger.exception("Crawl ingest failed for %s", page.url)
                    err_result = await session.execute(
                        select(IngestionSource).where(
                            IngestionSource.model_id == model_id,
                            IngestionSource.source_identifier == page.url,
                        )
                    )
                    err_src = err_result.scalar_one_or_none()
                    if err_src:
                        err_src.status = "failed"
                        await session.commit()


@router.post(
    "/models/{slug}/sources/crawl",
    response_model=CrawlResponse,
    status_code=202,
)
async def crawl_site_endpoint(
    body: CrawlRequest,
    model: RagModel = Depends(require_model_auth),
):
    """Crawl a website and ingest all discovered pages. Runs in the background."""
    asyncio.create_task(_crawl_site_background(model.id, body))

    return CrawlResponse(
        status="pending",
        message=f"Crawling {body.url} (max {body.max_pages} pages, depth {body.max_depth})",
        pages_queued=0,
    )
