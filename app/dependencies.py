from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import bcrypt
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.model_api_key import ModelApiKey
from app.models.rag_model import RagModel

logger = logging.getLogger("ragr.auth")

# ---------------------------------------------------------------------------
# Clerk JWT verification (lazy-initialised)
# ---------------------------------------------------------------------------

_clerk_client = None


def _get_clerk():
    global _clerk_client
    if _clerk_client is None and settings.clerk_secret_key:
        from clerk_backend_api import Clerk
        _clerk_client = Clerk(bearer_auth=settings.clerk_secret_key)
    return _clerk_client


@dataclass
class ClerkUser:
    user_id: str
    email: str | None = None

    @property
    def is_superuser(self) -> bool:
        return bool(settings.superuser_id and self.user_id == settings.superuser_id)


def _extract_bearer(authorization: str | None) -> str | None:
    """Extract token from 'Bearer <token>' header. Returns None if missing/malformed."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization.removeprefix("Bearer ")


async def _verify_clerk_token(request: Request) -> ClerkUser | None:
    """Verify a Clerk session token. Returns ClerkUser or None if Clerk is not configured or token is invalid."""
    clerk = _get_clerk()
    if clerk is None:
        return None

    try:
        from clerk_backend_api.security.types import AuthenticateRequestOptions
        request_state = clerk.authenticate_request(
            request,
            AuthenticateRequestOptions(
                authorized_parties=settings.console_origins,
            ),
        )
        if not request_state.is_signed_in:
            return None

        payload = request_state.payload or {}

        return ClerkUser(
            user_id=payload.get("sub", ""),
            email=payload.get("email"),
        )
    except Exception:
        logger.error("Clerk token verification failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Per-model API key validation
# ---------------------------------------------------------------------------


async def _validate_model_key(session: AsyncSession, model: RagModel, token: str) -> bool:
    """Check a token against per-model API keys. Returns True if valid."""
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


# ---------------------------------------------------------------------------
# Model slug resolvers
# ---------------------------------------------------------------------------


async def get_model_by_slug(
    slug: str,
    session: AsyncSession = Depends(get_session),
) -> RagModel:
    """Resolve a model by slug, raising 404 if not found or deleted."""
    result = await session.execute(select(RagModel).where(RagModel.slug == slug, RagModel.deleted_at.is_(None)))
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
        select(RagModel).where(RagModel.slug == slug, RagModel.is_active.is_(True), RagModel.deleted_at.is_(None))
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------


async def get_clerk_user(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> ClerkUser:
    """Extract and verify Clerk JWT. Raises 401 if not authenticated."""
    user = await _verify_clerk_token(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def require_model_auth(
    slug: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    authorization: str = Header(...),
) -> RagModel:
    """Authenticate with Clerk JWT, admin key, or per-model API key.

    For Clerk auth, verifies the user owns the model.
    Use this for model-scoped endpoints (sources, stats, conversations, api-keys).
    """
    model = await get_model_by_slug(slug, session)
    token = _extract_bearer(authorization)

    # Try per-model API key first (ragr_ prefix)
    if token and token.startswith("ragr_"):
        if await _validate_model_key(session, model, token):
            return model
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Try Clerk JWT
    clerk_user = await _verify_clerk_token(request)
    if clerk_user is not None:
        # Model has no owner — any authenticated user can access (legacy models)
        if model.owner_id is None:
            return model
        # Verify ownership
        if model.owner_id == clerk_user.user_id:
            return model
        # Superuser gets read-only access to all models
        if clerk_user.is_superuser and request.method in ("GET", "HEAD", "OPTIONS"):
            return model
        raise HTTPException(status_code=403, detail="You do not own this model")

    raise HTTPException(status_code=401, detail="Authentication required")


async def require_chat_auth(
    slug: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    authorization: Optional[str] = Header(None),
) -> RagModel:
    """Authenticate for chat access.

    - If the model has hosted_chat=True, no auth is required.
    - Otherwise, a valid Clerk JWT, admin key, or per-model API key is required.
    """
    model = await get_active_model_by_slug(slug, session)

    if model.hosted_chat and authorization is None:
        return model

    if authorization is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = _extract_bearer(authorization)

    # Per-model key
    if token and token.startswith("ragr_"):
        if await _validate_model_key(session, model, token):
            return model
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Clerk JWT
    clerk_user = await _verify_clerk_token(request)
    if clerk_user is not None:
        return model  # Any authenticated Clerk user can chat

    raise HTTPException(status_code=401, detail="Authentication required")
