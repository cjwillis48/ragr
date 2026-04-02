"""Integration test fixtures.

Uses a real PostgreSQL + pgvector database. Requires DATABASE_URL pointing
at a test database (e.g. ragr_test). Runs alembic migrations once per session.

Clerk auth is adaptive:
  - If CLERK_SECRET_KEY is a real dev-instance key: uses Backend API to create
    a session token for a test user (full JWT verification path).
  - If CLERK_SECRET_KEY is missing or a dummy value: falls back to
    dependency_overrides with a fake ClerkUser.

Anthropic is always mocked (non-deterministic, costs money).
Voyage is adaptive (real if VOYAGE_API_KEY is a real key).
"""

import os
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from fastapi import Depends, HTTPException
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database as db_module
import app.services.generation as gen_module
import app.services.embedder as embedder_module
import app.services.reranker as reranker_module
from app.database import get_session
from app.dependencies import (
    get_clerk_user, require_model_auth, require_chat_auth,
    get_model_by_slug, get_active_model_by_slug, ClerkUser,
)


logger = logging.getLogger("ragr.test")


# ---------------------------------------------------------------------------
# Run alembic migrations once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def run_migrations():
    """Run all alembic migrations against the test database."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"].replace("+asyncpg", ""))
    command.upgrade(alembic_cfg, "head")
    yield


# ---------------------------------------------------------------------------
# Per-test engine + session (fresh per event loop to avoid "attached to a
# different loop" errors with pytest-asyncio mode=auto)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_engine():
    """Create a fresh async engine for this test's event loop."""
    engine = create_async_engine(os.environ["DATABASE_URL"], echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Provide a DB session wrapped in a transaction that rolls back after the test."""
    async with db_engine.connect() as conn:
        txn = await conn.begin()
        # Use a nested savepoint so that session.commit() inside endpoint
        # handlers commits the savepoint, not the real transaction.
        # This lets us rollback the outer txn at test end.
        nested = await conn.begin_nested()
        session = AsyncSession(bind=conn, expire_on_commit=False)

        @event.listens_for(session.sync_session, "after_transaction_end")
        def restart_savepoint(session, transaction):
            """Re-open a savepoint after each commit so subsequent operations work."""
            if transaction.nested and not transaction._parent.nested:
                session.begin_nested()

        yield session

        await session.close()
        await txn.rollback()


# ---------------------------------------------------------------------------
# Override app's get_session AND async_session to use the test session
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def app(db_session):
    """Create the FastAPI app with test overrides."""
    from app.main import app as _app  # lazy: triggers full app initialization

    async def _override_get_session():
        yield db_session

    # Override DI-injected sessions (endpoints)
    _app.dependency_overrides[get_session] = _override_get_session

    # Override async_session factory (used by background tasks, streaming)
    # so they also use our test connection/transaction
    original_async_session = db_module.async_session

    class _TestSessionCtx:
        async def __aenter__(self):
            return db_session
        async def __aexit__(self, *args):
            pass

    db_module.async_session = lambda: _TestSessionCtx()

    yield _app

    _app.dependency_overrides.clear()
    db_module.async_session = original_async_session


# ---------------------------------------------------------------------------
# Clerk auth — adaptive real/mock
# ---------------------------------------------------------------------------

_DUMMY_KEYS = {"", "test-clerk-key", "test-key"}


def _is_real_clerk_key() -> bool:
    key = os.environ.get("CLERK_SECRET_KEY", "")
    return bool(key and key not in _DUMMY_KEYS)


_test_user_id: str | None = None


def _get_real_clerk_token() -> tuple[str, str]:
    """Create a test user + session via Clerk Backend API. Returns (token, user_id)."""
    global _test_user_id
    from clerk_backend_api import Clerk

    clerk = Clerk(bearer_auth=os.environ["CLERK_SECRET_KEY"])

    # Find or create a test user
    external_id = "ragr-integration-test"
    users = clerk.users.list(request={"external_id": [external_id]})
    if users:
        user = users[0]
    else:
        user = clerk.users.create(
            external_id=external_id,
            first_name="Integration",
            last_name="Test",
            email_address=["ragr-integration-test@example.com"],
            skip_password_requirement=True,
        )

    _test_user_id = user.id

    # Create session + token
    session = clerk.sessions.create(request={"user_id": user.id})
    token_resp = clerk.sessions.create_token(session_id=session.id)
    return token_resp.jwt, user.id


@pytest.fixture(scope="session")
def auth_mode():
    """Returns 'real' or 'mock' depending on Clerk key availability."""
    return "real" if _is_real_clerk_key() else "mock"


@pytest.fixture(scope="session")
def clerk_token_and_user_id(auth_mode):
    """Session-scoped: get Clerk JWT + user_id (real or mock)."""
    if auth_mode == "real":
        token, user_id = _get_real_clerk_token()
        return token, user_id
    return "mock-token", "test-user-id"


@pytest_asyncio.fixture
async def auth_headers(app, auth_mode, clerk_token_and_user_id):
    """Per-test: returns Authorization headers.

    In real-Clerk mode: no dependency overrides — the real authenticate_request()
    flow runs, verifying the JWT created by the Backend API. This tests the full
    auth chain end-to-end.

    In mock mode: overrides all auth dependencies so tests run without Clerk.
    """
    token, user_id = clerk_token_and_user_id

    if auth_mode == "mock":
        test_user = ClerkUser(user_id=user_id, email="test@example.com")
        app.dependency_overrides[get_clerk_user] = lambda: test_user

        async def _mock_require_model_auth(
            slug: str,
            session: AsyncSession = Depends(get_session),
        ):
            model = await get_model_by_slug(slug, session)
            if model.owner_id != user_id:
                raise HTTPException(status_code=403, detail="You do not own this model")
            return model

        async def _mock_require_chat_auth(
            slug: str,
            session: AsyncSession = Depends(get_session),
        ):
            return await get_active_model_by_slug(slug, session)

        app.dependency_overrides[require_model_auth] = _mock_require_model_auth
        app.dependency_overrides[require_chat_auth] = _mock_require_chat_auth

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
def test_user_id(clerk_token_and_user_id) -> str:
    """The user_id used for auth (real Clerk user ID or mock)."""
    return clerk_token_and_user_id[1]


# ---------------------------------------------------------------------------
# Mock Anthropic (always mocked)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Mock Anthropic client — always, since generation is non-deterministic."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='This is a test answer.\n<meta status="answered" />')]
    mock_response.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    # Mock the stream path too
    mock_stream_ctx = MagicMock()
    mock_final = MagicMock()
    mock_final.usage = mock_response.usage

    async def _mock_text_stream():
        yield "This is a test answer."
        yield '\n<meta status="answered" />'

    mock_stream_ctx.text_stream = _mock_text_stream()
    mock_stream_ctx.get_final_message = AsyncMock(return_value=mock_final)
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
    monkeypatch.setattr(gen_module, "_clients", MagicMock(get=MagicMock(return_value=mock_client)))


# ---------------------------------------------------------------------------
# Mock Voyage (adaptive — mock when using dummy key)
# ---------------------------------------------------------------------------

_DUMMY_VOYAGE_KEYS = {"", "test-voyage-key", "test-key"}


def _is_real_voyage_key() -> bool:
    key = os.environ.get("VOYAGE_API_KEY", "")
    return bool(key and key not in _DUMMY_VOYAGE_KEYS)


@pytest.fixture(autouse=True)
def mock_voyage(monkeypatch):
    """Mock Voyage embedder + reranker when using a dummy API key."""
    if _is_real_voyage_key():
        yield
        return

    # Mock embedder: return deterministic 1024-dim vectors
    import random
    rng = random.Random(42)

    mock_embed_client = MagicMock()

    async def _mock_embed(texts, model=None, input_type=None):
        result = MagicMock()
        result.embeddings = [[rng.gauss(0, 1) for _ in range(1024)] for _ in texts]
        result.total_tokens = len(texts) * 10
        return result

    mock_embed_client.embed = _mock_embed
    monkeypatch.setattr(embedder_module, "_clients", MagicMock(get=MagicMock(return_value=mock_embed_client)))

    # Mock reranker: return indices in input order with descending scores
    mock_rerank_client = MagicMock()

    async def _mock_rerank(query, documents, model=None, top_k=5):
        n = min(top_k, len(documents))
        result = MagicMock()
        result.results = [
            MagicMock(index=i, relevance_score=1.0 - i * 0.1)
            for i in range(n)
        ]
        result.total_tokens = len(documents) * 5
        return result

    mock_rerank_client.rerank = _mock_rerank
    monkeypatch.setattr(reranker_module, "_clients", MagicMock(get=MagicMock(return_value=mock_rerank_client)))

    yield


# ---------------------------------------------------------------------------
# Async HTTP client
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(app, auth_headers):
    """Async HTTP client for making requests to the app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        c.headers.update(auth_headers)
        yield c


# ---------------------------------------------------------------------------
# Cleanup: truncate all tables after the full session
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session", autouse=True)
async def cleanup_after_session():
    """Truncate all tables after the test session."""
    yield
    engine = create_async_engine(os.environ["DATABASE_URL"], echo=False)
    async with engine.begin() as conn:
        await conn.execute(text(
            "TRUNCATE rag_models, content_chunks, conversations, messages, "
            "ingestion_sources, model_api_keys, token_usage, system_prompt_history "
            "CASCADE"
        ))
    await engine.dispose()
