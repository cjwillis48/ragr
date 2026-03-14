from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_session
from app.dependencies import require_model_auth
from app.models.content import ContentChunk
from app.models.conversation import Conversation, Message
from app.models.ingestion_source import IngestionSource
from app.models.rag_model import RagModel
from app.schemas.admin import ChunkResponse, ConversationDetailResponse, ConversationListResponse, ConversationSummaryResponse, MessageResponse, StatsResponse
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
    chunk_count = await session.scalar(
        select(func.count()).select_from(ContentChunk).where(ContentChunk.model_id == model.id)
    )

    convo_count = await session.scalar(
        select(func.count()).select_from(Conversation).where(Conversation.model_id == model.id)
    )

    message_count = await session.scalar(
        select(func.count()).select_from(Message).where(Message.model_id == model.id)
    )

    unanswered = await session.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.model_id == model.id, Message.status == "unanswered")
    )

    source_count = await session.scalar(
        select(func.count()).select_from(IngestionSource).where(IngestionSource.model_id == model.id)
    )

    usage = await get_current_month_usage(session, model)
    current_cost = usage.estimated_cost if usage else 0.0

    return StatsResponse(
        model_slug=model.slug,
        total_chunks=chunk_count or 0,
        total_conversations=convo_count or 0,
        total_messages=message_count or 0,
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
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List conversations for a model, newest first."""
    base = select(Conversation).where(Conversation.model_id == model.id)

    total = await session.scalar(
        select(func.count()).select_from(base.subquery())
    )

    result = await session.execute(
        base.order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    convos = result.scalars().all()

    return ConversationListResponse(
        model_slug=model.slug,
        conversations=[ConversationSummaryResponse.model_validate(c) for c in convos],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/models/{slug}/conversations/{conversation_id}/messages",
    response_model=ConversationDetailResponse,
)
async def get_conversation_messages(
    conversation_id: int,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Get all messages for a specific conversation."""
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.model_id == model.id)
        .options(selectinload(Conversation.messages))
    )
    convo = result.scalar_one_or_none()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse.model_validate(convo)


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
