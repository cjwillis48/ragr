from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.cors import sync_origins
from app.database import get_session
from app.dependencies import (
    ClerkUser,
    get_active_model_by_slug,
    get_model_by_slug,
    get_clerk_user,
    require_model_auth,
)
from app.models.rag_model import RagModel
from app.models.system_prompt_history import SystemPromptHistory
from app.schemas.models import ChatTheme, RagModelCreate, RagModelPublic, RagModelRead, RagModelUpdate
from app.services.budget import check_budget

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=RagModelRead, status_code=201)
async def create_model(
    body: RagModelCreate,
    clerk_user: ClerkUser = Depends(get_clerk_user),
    session: AsyncSession = Depends(get_session),
):
    """Create a new RAG model."""
    existing = await session.execute(select(RagModel).where(RagModel.slug == body.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Model with this slug already exists")

    model = RagModel(
        owner_id=clerk_user.user_id,
        name=body.name,
        slug=body.slug,
        description=body.description,
        system_prompt=body.system_prompt,
        chat_theme=body.chat_theme.model_dump(exclude_none=True) if body.chat_theme else None,
        chunk_size=body.chunk_size if body.chunk_size is not None else settings.default_chunk_size,
        chunk_overlap=body.chunk_overlap if body.chunk_overlap is not None else settings.default_chunk_overlap,
        similarity_threshold=body.similarity_threshold if body.similarity_threshold is not None else settings.default_similarity_threshold,
        top_k=body.top_k if body.top_k is not None else settings.default_top_k,
        embedding_model=body.embedding_model or settings.default_embedding_model,
        generation_model=body.generation_model or settings.default_generation_model,
        reranker_enabled=body.reranker_enabled if body.reranker_enabled is not None else settings.default_reranker_enabled,
        rerank_model=body.rerank_model or settings.default_rerank_model,
        history_turns=body.history_turns if body.history_turns is not None else settings.default_history_turns,
        hosted_chat=body.hosted_chat if body.hosted_chat is not None else settings.default_hosted_chat,
        allowed_origins=body.allowed_origins if body.allowed_origins is not None else [],
        budget_limit=body.budget_limit if body.budget_limit is not None else settings.default_budget_limit,
        custom_anthropic_key=body.custom_anthropic_key,
        custom_voyage_key=body.custom_voyage_key,
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    if model.allowed_origins:
        await sync_origins(session)
    return RagModelRead.from_model(model)


@router.get("", response_model=list[RagModelRead])
async def list_models(
    clerk_user: ClerkUser = Depends(get_clerk_user),
    session: AsyncSession = Depends(get_session),
):
    """List all RAG models. Clerk users see only their own models."""
    query = select(RagModel).where(RagModel.deleted_at.is_(None)).order_by(RagModel.created_at)

    if clerk_user and not clerk_user.is_superuser:
        query = query.where(RagModel.owner_id == clerk_user.user_id)

    result = await session.execute(query)
    return [RagModelRead.from_model(m) for m in result.scalars().all()]


@router.get("/{slug}/info", response_model=RagModelPublic)
async def get_model_public(
    model: RagModel = Depends(get_active_model_by_slug),
    session: AsyncSession = Depends(get_session),
):
    """Public model info for the chat UI — no auth required."""
    accepting = await check_budget(session, model)
    info = RagModelPublic.model_validate(model)
    info.accepting_requests = accepting
    return info


@router.get("/{slug}", response_model=RagModelRead)
async def get_model(model: RagModel = Depends(require_model_auth)):
    """Get a RAG model by slug."""
    return RagModelRead.from_model(model)


@router.patch("/{slug}", response_model=RagModelRead)
async def update_model(
    body: RagModelUpdate,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Update a RAG model's configuration."""
    update_data = body.model_dump(exclude_unset=True)

    # Only allow setting known mutable fields — prevents schema drift from writing unexpected attributes
    _MUTABLE_FIELDS = {
        "name", "description", "system_prompt", "chat_theme",
        "chunk_size", "chunk_overlap", "similarity_threshold", "top_k",
        "embedding_model", "generation_model",
        "reranker_enabled", "rerank_model", "rerank_candidates", "rerank_threshold",
        "keyword_search_enabled", "sample_questions",
        "history_turns", "hosted_chat", "allowed_origins", "budget_limit",
        "is_active", "custom_anthropic_key", "custom_voyage_key",
    }
    update_data = {k: v for k, v in update_data.items() if k in _MUTABLE_FIELDS}

    # Record system prompt change in history
    if "system_prompt" in update_data and update_data["system_prompt"] != model.system_prompt:
        session.add(SystemPromptHistory(
            model_id=model.id,
            prompt_text=update_data["system_prompt"],
            source="manual",
        ))

    # Serialize chat_theme Pydantic model to dict for JSONB column
    if "chat_theme" in update_data and update_data["chat_theme"] is not None:
        update_data["chat_theme"] = update_data["chat_theme"].model_dump(exclude_none=True)

    for field, value in update_data.items():
        setattr(model, field, value)
    await session.commit()
    await session.refresh(model)
    if "allowed_origins" in update_data:
        await sync_origins(session)
    return RagModelRead.from_model(model)


@router.delete("/{slug}", status_code=204)
async def delete_model(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Soft-delete a RAG model. Data is preserved but hidden from all queries."""
    model.deleted_at = func.now()
    await session.commit()
    await sync_origins(session)


@router.get("/{slug}/theme", response_model=ChatTheme)
async def get_theme(model: RagModel = Depends(get_active_model_by_slug)):
    """Public endpoint — returns the chat widget theme for embedding."""
    return ChatTheme(**(model.chat_theme or {}))


@router.patch("/{slug}/theme", response_model=ChatTheme)
async def update_theme(
    body: ChatTheme,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Update chat widget theme. Merges with existing theme — only sent fields are updated."""
    merged = {**(model.chat_theme or {}), **body.model_dump(exclude_unset=True)}
    model.chat_theme = merged
    await session.commit()
    await session.refresh(model)
    return ChatTheme(**model.chat_theme)
