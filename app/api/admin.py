from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.dependencies import require_model_auth
from app.models.content import ContentChunk
from app.models.conversation import ConversationLog
from app.models.ingestion_source import IngestionSource
from app.models.rag_model import RagModel
from app.schemas.admin import ChunkResponse, ConversationListResponse, ConversationResponse, StatsResponse
from app.services.budget import get_current_month_usage

router = APIRouter(tags=["admin"])


@router.get(
    "/models/{slug}/stats",
    response_model=StatsResponse,
)
async def model_stats(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Get per-model statistics."""
    # Total chunks
    chunk_count = await session.scalar(
        select(func.count()).select_from(ContentChunk).where(ContentChunk.model_id == model.id)
    )

    # Total conversations
    convo_count = await session.scalar(
        select(func.count()).select_from(ConversationLog).where(ConversationLog.model_id == model.id)
    )

    # Unanswered questions
    unanswered = await session.scalar(
        select(func.count())
        .select_from(ConversationLog)
        .where(ConversationLog.model_id == model.id, ConversationLog.status == "unanswered")
    )

    # Total sources
    source_count = await session.scalar(
        select(func.count()).select_from(IngestionSource).where(IngestionSource.model_id == model.id)
    )

    # Current month cost
    usage = await get_current_month_usage(session, model)
    current_cost = usage.estimated_cost if usage else 0.0

    return StatsResponse(
        model_slug=model.slug,
        total_chunks=chunk_count or 0,
        total_conversations=convo_count or 0,
        unanswered_questions=unanswered or 0,
        current_month_cost=round(current_cost, 4),
        budget_limit=model.budget_limit,
        budget_remaining=round(model.budget_limit - current_cost, 4),
        total_sources=source_count or 0,
    )


@router.get(
    "/models/{slug}/conversations",
    response_model=ConversationListResponse,
)
async def list_conversations(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
    status: str | None = Query(None, description="Filter by status: answered, unanswered, off_topic"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List conversations for a model, newest first. Filterable by status."""
    base = select(ConversationLog).where(ConversationLog.model_id == model.id)
    if status:
        base = base.where(ConversationLog.status == status)

    total = await session.scalar(
        select(func.count()).select_from(base.subquery())
    )

    result = await session.execute(
        base.order_by(ConversationLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    convos = result.scalars().all()

    return ConversationListResponse(
        model_slug=model.slug,
        conversations=[ConversationResponse.model_validate(c) for c in convos],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/models/{slug}/chunks",
    response_model=list[ChunkResponse],
)
async def get_chunks(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
    ids: str = Query(..., description="Comma-separated chunk IDs"),
):
    """Fetch chunks by ID for a model. Used to inspect retrieved context for a conversation."""
    try:
        chunk_ids = [int(i) for i in ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be comma-separated integers")

    if len(chunk_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 chunk IDs per request")

    result = await session.execute(
        select(ContentChunk).where(
            ContentChunk.model_id == model.id,
            ContentChunk.id.in_(chunk_ids),
        )
    )
    return result.scalars().all()
