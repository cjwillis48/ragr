from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.tenancy import SET_TENANT, current_model_id

engine = None
async_session = None


@event.listens_for(Session, "after_begin")
def _apply_tenant(session: Session, transaction, connection) -> None:
    """Publish the current tenant to Postgres for row-level security.

    Fires per transaction rather than per session on purpose: the setting is
    transaction-local, and several request paths commit part-way through (see
    `_log_message` in app/api/chat.py), which would otherwise drop tenant scoping
    for everything after the commit.

    This only covers tenants known *before* the transaction opens — the worker,
    and any session reopened after a commit. A request that discovers its tenant
    by querying for it has already begun a transaction by then; that case calls
    `bind_tenant` instead.

    Skipping when there is no tenant costs nothing in safety: the setting is
    transaction-local, so a transaction that never sets it reads NULL rather than
    whatever the previous one on that pooled connection used, and the policy
    treats NULL as "match nothing". It does save a round trip on every
    tenant-less transaction — health checks, and the worker's queue poll every
    1.5 seconds.

    Must use `connection` and not `session`: the Session is mid-provisioning
    while this fires.
    """
    model_id = current_model_id()
    if model_id is None:
        return
    connection.execute(SET_TENANT, {"model_id": str(model_id)})


def _init_engine():
    global engine, async_session
    if engine is None:
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL must be set")
        engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    _init_engine()
    async with async_session() as session:
        yield session
