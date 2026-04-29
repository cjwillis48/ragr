import logging
from pathlib import Path
import time

import pymupdf  # noqa: F401 — eager import to avoid cold-start delay

from fastapi import APIRouter, Depends, HTTPException, Query, Response, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.dependencies import require_model_auth
from app.models.content import ContentChunk
from app.models.ingestion_job import IngestionJob
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
from app.services.html import strip_html
from app.services.ingest import ingest_content
from app.services.r2 import is_configured as r2_is_configured
from app.services.rate_limit import RateLimiter
from app.services.url_validation import SSRFError, validate_url
from app.services.wikipedia import is_wikipedia_url

router = APIRouter(tags=["sources"])
logger = logging.getLogger("ragr.sources")

_ingest_limiter = RateLimiter(max_requests=20, window_seconds=60)

ALLOWED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf", ".csv", ".json"}


class ExtractionError(Exception):
    """Raised when text extraction from a file fails."""



@router.get(
    "/models/{slug}/sources",
    response_model=SourceListResponse,
)
async def list_sources(
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        search: str | None = Query(None, max_length=200, description="Filter sources by identifier or URL"),
):
    """List ingested sources for a model with pagination and optional search."""
    base_filter = [IngestionSource.model_id == model.id]
    if search:
        pattern = f"%{search}%"
        base_filter.append(
            IngestionSource.source_identifier.ilike(pattern) | IngestionSource.source_url.ilike(pattern)
        )

    count_result = await session.execute(
        select(func.count()).select_from(IngestionSource).where(*base_filter)
    )
    total = count_result.scalar_one()

    result = await session.execute(
        select(IngestionSource)
        .where(*base_filter)
        .order_by(IngestionSource.ingested_at.desc())
        .limit(limit)
        .offset(offset)
    )
    sources = result.scalars().all()
    return SourceListResponse(
        model_slug=model.slug,
        sources=[SourceResponse.model_validate(s) for s in sources],
        total=total,
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
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
):
    """List content chunks for a source with pagination. Useful for debugging ingestion."""
    result = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.id == source_id,
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    base_query = select(ContentChunk).where(
        ContentChunk.model_id == model.id,
        ContentChunk.source_identifier == source.source_identifier,
    )

    total = await session.scalar(
        select(func.count()).select_from(base_query.subquery())
    )

    chunks_result = await session.execute(
        base_query.order_by(ContentChunk.id).limit(limit).offset(offset)
    )
    chunks = chunks_result.scalars().all()
    return ChunkListResponse(
        source_identifier=source.source_identifier,
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
        total=total or 0,
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
    if not _ingest_limiter.is_allowed(f"ingest:{model.id}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")
    has_content = body.content is not None
    has_url = body.url is not None
    has_urls = body.urls is not None and len(body.urls) > 0

    if not (has_content or has_url or has_urls):
        raise HTTPException(status_code=422, detail="Provide one of 'content', 'url', or 'urls'")

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

    # Validate URLs before processing (skip Wikipedia — known-safe, uses API)
    url_list = body.urls if has_urls else [body.url]
    for u in url_list:
        if not is_wikipedia_url(u):
            try:
                await validate_url(u)
            except SSRFError as e:
                raise HTTPException(status_code=422, detail=str(e))

    # Normalise single url into a list; source_identifier only applies to single url
    urls = url_list
    source_ids = list(urls) if has_urls else [body.source_identifier or body.url]

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

        session.add(IngestionJob(
            model_id=model.id,
            job_type="url",
            job_params={"url": url, "source_identifier": source_id},
        ))
        results.append(CreateSourceResponse(
            source_identifier=source_id,
            status="pending",
            message=f"URL ingestion started for {url}",
        ))

    await session.commit()

    response.status_code = 202
    return results


def _extract_text(filename: str, raw: bytes) -> tuple[str, str]:
    """Extract text and content_type from raw file bytes. Raises ExtractionError on failure."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ExtractionError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        try:
            doc = pymupdf.Document(stream=raw, filetype="pdf")
        except Exception:
            raise ExtractionError("Could not parse PDF")
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(pages)
        page_count = len(pages)
        char_count = len(text.strip())
        logger.info("pdf_extracted", extra={"filename": filename, "pages": page_count, "chars": char_count})
        if char_count < 100:
            raise ExtractionError("PDF appears to be scanned/image-based")
        return text, "pdf"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ExtractionError("File must be UTF-8 encoded text")

    if ext in (".html", ".htm"):
        return strip_html(text), "html"
    elif ext == ".md":
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
    if not _ingest_limiter.is_allowed(f"ingest:{model.id}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")

    from app.config import settings

    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=422,
            detail=f"Too many files: max {settings.max_upload_files} per request",
        )

    prepared_files: list[tuple[str, str, str]] = []
    results = []

    t0 = time.monotonic()
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required for all files")

        # Check file size before reading into memory
        if file.size and file.size > max_bytes:
            raise HTTPException(
                status_code=422,
                detail=f"File '{file.filename}' exceeds {settings.max_upload_size_mb}MB limit",
            )
        raw = await file.read()
        if len(raw) > max_bytes:
            raise HTTPException(
                status_code=422,
                detail=f"File '{file.filename}' exceeds {settings.max_upload_size_mb}MB limit",
            )
        t_read = time.monotonic()
        try:
            text, content_type = _extract_text(file.filename, raw)
        except ExtractionError as e:
            raise HTTPException(status_code=422, detail=str(e))
        t_extract = time.monotonic()
        logger.info("upload_extracted", extra={
            "filename": file.filename, "read_ms": round((t_read - t0) * 1000, 1),
            "extract_ms": round((t_extract - t_read) * 1000, 1), "size_kb": len(raw) // 1024,
        })
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
            existing.raw_content = text
        else:
            session.add(IngestionSource(
                model_id=model.id,
                source_identifier=filename,
                content_hash="",
                chunk_count=0,
                source_url=filename,
                content_type=content_type,
                status="pending",
                raw_content=text,
            ))

        session.add(IngestionJob(
            model_id=model.id,
            job_type="file",
            job_params={"source_identifier": filename, "content_type": content_type},
        ))
        results.append(CreateSourceResponse(
            source_identifier=filename,
            status="pending",
            message=f"Ingestion started for {filename}",
        ))

    await session.commit()
    logger.info("upload_db_upserts",
                extra={"duration_ms": round((time.monotonic() - t_db) * 1000, 1), "files": len(prepared_files)})

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
        url = await generate_presigned_upload_url(object_key, f.content_type)
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
        # Reject path traversal attempts
        if ".." in f.object_key:
            raise HTTPException(status_code=422, detail=f"Invalid object key: {f.object_key}")
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

        session.add(IngestionJob(
            model_id=model.id,
            job_type="r2_file",
            job_params={"object_key": f.object_key, "filename": f.filename},
        ))
        results.append(CreateSourceResponse(
            source_identifier=f.filename,
            status="pending",
            message=f"Ingestion started for {f.filename}",
        ))

    await session.commit()
    return results



@router.post(
    "/models/{slug}/sources/crawl",
    response_model=CrawlResponse,
    status_code=202,
)
async def crawl_site_endpoint(
        body: CrawlRequest,
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
):
    """Crawl a website and ingest all discovered pages. Runs in the background."""
    if not _ingest_limiter.is_allowed(f"ingest:{model.id}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")
    if not is_wikipedia_url(body.url):
        try:
            await validate_url(body.url)
        except SSRFError as e:
            raise HTTPException(status_code=422, detail=str(e))

    # Create a crawling source immediately so the UI shows activity
    existing = await session.execute(
        select(IngestionSource).where(
            IngestionSource.model_id == model.id,
            IngestionSource.source_identifier == body.url,
        )
    )
    src = existing.scalar_one_or_none()
    if src:
        src.status = "crawling"
    else:
        session.add(IngestionSource(
            model_id=model.id,
            source_identifier=body.url,
            content_hash="",
            chunk_count=0,
            source_url=body.url,
            content_type="html",
            status="crawling",
        ))
    session.add(IngestionJob(
        model_id=model.id,
        job_type="crawl",
        job_params={
            "url": body.url,
            "max_pages": body.max_pages,
            "max_depth": body.max_depth,
            "prefix": body.prefix,
            "exclude_patterns": body.exclude_patterns,
        },
    ))
    await session.commit()

    return CrawlResponse(
        status="pending",
        message=f"Crawling {body.url} (max {body.max_pages} pages, depth {body.max_depth})",
        pages_queued=0,
    )
