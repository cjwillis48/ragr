from typing import Optional

import bcrypt
from fastapi import Depends, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.model_api_key import ModelApiKey
from app.models.rag_model import RagModel


async def require_api_key(authorization: str = Header(...)) -> None:
    """Validate admin API key from Authorization: Bearer <key> header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ")
    if token != settings.ragr_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _validate_model_key(session: AsyncSession, model: RagModel, token: str) -> bool:
    """Check a token against the admin key or per-model API keys. Returns True if valid."""
    if token == settings.ragr_api_key:
        return True

    key_prefix = token[:12]
    result = await session.execute(
        select(ModelApiKey).where(
            ModelApiKey.model_id == model.id,
            ModelApiKey.is_active.is_(True),
            ModelApiKey.key_prefix == key_prefix,
        )
    )
    candidate = result.scalar_one_or_none()
    if candidate and bcrypt.checkpw(token.encode(), candidate.key_hash.encode()):
        candidate.last_used_at = func.now()
        await session.flush()
        return True

    return False


async def get_model_by_slug(
    slug: str,
    session: AsyncSession = Depends(get_session),
) -> RagModel:
    """Resolve a model by slug, raising 404 if not found."""
    result = await session.execute(select(RagModel).where(RagModel.slug == slug))
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


async def get_active_model_by_slug(
    slug: str,
    session: AsyncSession = Depends(get_session),
) -> RagModel:
    """Resolve an active model by slug, raising 404 if not found or inactive."""
    result = await session.execute(
        select(RagModel).where(RagModel.slug == slug, RagModel.is_active.is_(True))
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


async def require_model_auth(
    slug: str,
    session: AsyncSession = Depends(get_session),
    authorization: str = Header(...),
) -> RagModel:
    """Authenticate with admin key or per-model API key. Returns the resolved model.

    Use this for model-scoped endpoints (sources, stats, conversations, api-keys).
    """
    model = await get_model_by_slug(slug, session)

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ")

    if await _validate_model_key(session, model, token):
        return model

    raise HTTPException(status_code=401, detail="Invalid API key")


async def require_chat_auth(
    slug: str,
    session: AsyncSession = Depends(get_session),
    authorization: Optional[str] = Header(None),
) -> RagModel:
    """Authenticate for chat access.

    - If the model has public_access=True, no auth is required.
    - Otherwise, a valid admin key or per-model API key is required.
    """
    model = await get_active_model_by_slug(slug, session)

    if model.public_access and authorization is None:
        return model

    if authorization is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ")

    if await _validate_model_key(session, model, token):
        return model

    raise HTTPException(status_code=401, detail="Invalid API key")