import secrets

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.dependencies import require_model_auth
from app.models.model_api_key import ModelApiKey
from app.models.rag_model import RagModel
from app.schemas.api_keys import ApiKeyCreate, ApiKeyCreateResponse, ApiKeyRead

router = APIRouter(tags=["api-keys"])

KEY_PREFIX = "ragr_"


def _generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


@router.post(
    "/models/{slug}/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=201,
)
async def create_api_key(
    body: ApiKeyCreate,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Generate a new API key for a model. The raw key is only returned once."""
    raw_key = _generate_key()
    key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()

    api_key = ModelApiKey(
        model_id=model.id,
        label=body.label,
        key_hash=key_hash,
        key_prefix=raw_key[:12],
    )
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)

    return ApiKeyCreateResponse(
        id=api_key.id,
        label=api_key.label,
        key_prefix=api_key.key_prefix,
        raw_key=raw_key,
        created_at=api_key.created_at,
    )


@router.get(
    "/models/{slug}/api-keys",
    response_model=list[ApiKeyRead],
)
async def list_api_keys(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """List all API keys for a model (hashes are never exposed)."""
    result = await session.execute(
        select(ModelApiKey)
        .where(ModelApiKey.model_id == model.id)
        .order_by(ModelApiKey.created_at.desc())
    )
    return result.scalars().all()


@router.delete(
    "/models/{slug}/api-keys/{key_id}",
    status_code=204,
)
async def revoke_api_key(
    key_id: int,
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Revoke an API key (soft delete)."""
    result = await session.execute(
        select(ModelApiKey).where(
            ModelApiKey.id == key_id,
            ModelApiKey.model_id == model.id,
        )
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = False
    await session.commit()