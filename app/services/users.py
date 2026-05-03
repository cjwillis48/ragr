from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


async def get_or_create_user(
    session: AsyncSession,
    clerk_user_id: str,
    email: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
) -> User:
    """Upsert a user row keyed by clerk_user_id.

    Refreshes email/first/last from Clerk-derived values on each call but
    never touches allow_global_keys (operator-controlled in pgweb).
    """
    insert_stmt = pg_insert(User).values(
        clerk_user_id=clerk_user_id,
        email=email,
        first_name=first_name,
        last_name=last_name,
    )
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["clerk_user_id"],
        set_={
            "email": insert_stmt.excluded.email,
            "first_name": insert_stmt.excluded.first_name,
            "last_name": insert_stmt.excluded.last_name,
            "updated_at": func.now(),
        },
        where=(
            User.email.is_distinct_from(insert_stmt.excluded.email)
            | User.first_name.is_distinct_from(insert_stmt.excluded.first_name)
            | User.last_name.is_distinct_from(insert_stmt.excluded.last_name)
        ),
    ).returning(User)

    result = await session.execute(upsert_stmt)
    user = result.scalar_one_or_none()
    if user is not None:
        await session.commit()
        return user

    # No row returned means the row exists and the WHERE filter found nothing to update.
    existing = await session.execute(select(User).where(User.clerk_user_id == clerk_user_id))
    return existing.scalar_one()


async def owner_can_use_global_keys(session: AsyncSession, owner_clerk_id: str) -> bool:
    """True iff a users row exists for this Clerk ID with allow_global_keys=true."""
    result = await session.execute(
        select(User.allow_global_keys).where(User.clerk_user_id == owner_clerk_id)
    )
    flag = result.scalar_one_or_none()
    return bool(flag)
