from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = None
async_session = None


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
