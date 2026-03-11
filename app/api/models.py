from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.cors import sync_origins
from app.database import get_session
from app.dependencies import get_active_model_by_slug, get_model_by_slug, require_api_key
from app.models.rag_model import RagModel
from app.schemas.models import RagModelCreate, RagModelPublic, RagModelRead, RagModelUpdate
from app.services.budget import check_budget

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=RagModelRead, status_code=201, dependencies=[Depends(require_api_key)])
async def create_model(body: RagModelCreate, session: AsyncSession = Depends(get_session)):
    """Create a new RAG model."""
    # Check slug uniqueness
    existing = await session.execute(select(RagModel).where(RagModel.slug == body.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Model with this slug already exists")

    model = RagModel(
        name=body.name,
        slug=body.slug,
        description=body.description,
        greeting=body.greeting,
        placeholder=body.placeholder,
        system_prompt=body.system_prompt,
        chunk_size=body.chunk_size if body.chunk_size is not None else settings.default_chunk_size,
        chunk_overlap=body.chunk_overlap if body.chunk_overlap is not None else settings.default_chunk_overlap,
        similarity_threshold=body.similarity_threshold if body.similarity_threshold is not None else settings.default_similarity_threshold,
        top_k=body.top_k if body.top_k is not None else settings.default_top_k,
        embedding_model=body.embedding_model or settings.default_embedding_model,
        generation_model=body.generation_model or settings.default_generation_model,
        reranker_enabled=body.reranker_enabled if body.reranker_enabled is not None else False,
        rerank_model=body.rerank_model or settings.default_rerank_model,
        public_access=body.public_access if body.public_access is not None else True,
        allowed_origins=body.allowed_origins if body.allowed_origins is not None else [],
        budget_limit=body.budget_limit if body.budget_limit is not None else settings.default_budget_limit,
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    if model.allowed_origins:
        await sync_origins(session)
    return model


@router.get("", response_model=list[RagModelRead], dependencies=[Depends(require_api_key)])
async def list_models(session: AsyncSession = Depends(get_session)):
    """List all RAG models."""
    result = await session.execute(select(RagModel).order_by(RagModel.created_at))
    return result.scalars().all()


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


@router.get("/{slug}", response_model=RagModelRead, dependencies=[Depends(require_api_key)])
async def get_model(model: RagModel = Depends(get_model_by_slug)):
    """Get a RAG model by slug."""
    return model


@router.patch("/{slug}", response_model=RagModelRead, dependencies=[Depends(require_api_key)])
async def update_model(
    body: RagModelUpdate,
    model: RagModel = Depends(get_model_by_slug),
    session: AsyncSession = Depends(get_session),
):
    """Update a RAG model's configuration."""
    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)
    await session.commit()
    await session.refresh(model)
    if "allowed_origins" in update_data:
        await sync_origins(session)
    return model


@router.delete("/{slug}", status_code=204, dependencies=[Depends(require_api_key)])
async def delete_model(
    model: RagModel = Depends(get_model_by_slug),
    session: AsyncSession = Depends(get_session),
):
    """Delete a RAG model and all associated data."""
    await session.delete(model)
    await session.commit()
    await sync_origins(session)
