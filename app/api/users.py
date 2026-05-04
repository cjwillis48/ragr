from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.dependencies import ClerkUser, get_clerk_user
from app.models.user import User

router = APIRouter(prefix="/users", tags=["users"])


class CurrentUser(BaseModel):
    user_id: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    allow_global_keys: bool


@router.get("/me", response_model=CurrentUser, include_in_schema=False)
async def get_me(
    clerk_user: ClerkUser = Depends(get_clerk_user),
    session: AsyncSession = Depends(get_session),
) -> CurrentUser:
    """Return the authenticated user's profile and allowlist status.

    Drives proactive UI: the console uses allow_global_keys to decide whether
    BYOK fields are required before the user submits and gets a 403.
    """
    result = await session.execute(select(User).where(User.clerk_user_id == clerk_user.user_id))
    db_user = result.scalar_one()
    return CurrentUser(
        user_id=db_user.clerk_user_id,
        email=db_user.email,
        first_name=db_user.first_name,
        last_name=db_user.last_name,
        allow_global_keys=db_user.allow_global_keys,
    )
