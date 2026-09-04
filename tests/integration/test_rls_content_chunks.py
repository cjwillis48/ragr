"""Row-level security on content_chunks, exercised as the restricted role.

These tests deliberately avoid the shared `db_session` fixture. That fixture runs
everything as the owner — which is the initdb superuser, and superusers bypass row
security unconditionally — inside one long-lived transaction on one connection.
Nothing about RLS is observable through it. So this module builds its own engine
and connects as `ragr_app`.

Fixtures are function-scoped rather than module-scoped because the surrounding
conftest gives each test its own event loop, and an engine outlives its loop badly.

Every query here omits `WHERE model_id`, on purpose. That is the whole point: the
question is what the database does when the application forgets.
"""

import os

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.content import ContentChunk
from app.models.rag_model import RagModel
from app.tenancy import _MODEL_ID, bind_tenant, tenant_scope

A_SECRET = "RLS_PROBE_TENANT_A"
B_SECRET = "RLS_PROBE_TENANT_B"
A_SLUG, B_SLUG = "rls-probe-a", "rls-probe-b"

# No model_id predicate anywhere — the policy is the only thing scoping this.
UNSCOPED = select(ContentChunk.content).where(
    ContentChunk.content.in_([A_SECRET, B_SECRET])
)


def _app_url() -> str:
    """The owner's URL with the restricted role's credentials swapped in."""
    if override := os.environ.get("TEST_APP_DATABASE_URL"):
        return override
    owner = os.environ["DATABASE_URL"]
    scheme, _, rest = owner.partition("://")
    _, _, hostpart = rest.partition("@")
    return f"{scheme}://ragr_app:ragr_app@{hostpart}"


@pytest.fixture
async def app_engine(run_migrations):
    """Engine connected as the restricted role, with LOGIN granted just-in-time.

    The migration creates `ragr_app` NOLOGIN and without a password, since
    credentials belong in a secret store rather than a schema. Tests supply a
    throwaway one here so `make test-integration` needs no extra setup.
    """
    owner = create_async_engine(os.environ["DATABASE_URL"], poolclass=None)
    try:
        async with owner.begin() as conn:
            await conn.execute(text("ALTER ROLE ragr_app WITH LOGIN PASSWORD 'ragr_app'"))
    except ProgrammingError as exc:
        pytest.skip(f"cannot grant ragr_app LOGIN: {exc}")
    finally:
        await owner.dispose()

    engine = create_async_engine(_app_url())
    yield engine
    await engine.dispose()


@pytest.fixture
async def tenants(run_migrations):
    """Two tenants with one chunk each, really committed so a second connection sees them."""
    owner = create_async_engine(os.environ["DATABASE_URL"])
    maker = async_sessionmaker(owner, class_=AsyncSession, expire_on_commit=False)
    ids: dict[str, int] = {}
    async with maker() as session:
        for slug, secret in ((A_SLUG, A_SECRET), (B_SLUG, B_SECRET)):
            model = RagModel(
                owner_id=f"rls-{slug}", name=slug, slug=slug, description="", system_prompt="",
                embedding_model="voyage-4-lite", generation_model="claude-haiku-4-5",
                sample_messages=[], allowed_origins=[],
            )
            session.add(model)
            await session.flush()
            session.add(ContentChunk(
                model_id=model.id, source_identifier=f"{slug}.md", source_url="",
                content=secret, content_type="text", metadata_={}, embedding=[0.1] * 1024,
            ))
            ids[slug] = model.id
        await session.commit()

    yield ids

    async with maker() as session:
        await session.execute(delete(ContentChunk).where(ContentChunk.model_id.in_(ids.values())))
        await session.execute(delete(RagModel).where(RagModel.id.in_(ids.values())))
        await session.commit()
    await owner.dispose()


@pytest.fixture
async def app_session(app_engine, tenants):
    async with async_sessionmaker(app_engine, class_=AsyncSession, expire_on_commit=False)() as s:
        yield s


async def _contents(session) -> list[str]:
    return sorted((await session.execute(UNSCOPED)).scalars().all())


class TestPolicy:
    async def test_no_tenant_returns_nothing(self, app_session, tenants):
        """The default is deny. A query with no tenant bound sees no rows at all."""
        assert await _contents(app_session) == []

    async def test_tenant_sees_only_its_own(self, app_session, tenants):
        with tenant_scope(tenants[A_SLUG]):
            assert await _contents(app_session) == [A_SECRET]

    async def test_owner_still_sees_everything(self, db_session, tenants):
        """Guards against a false pass above: the rows really are there.

        The owner is the initdb superuser, and superusers bypass row security
        outright — FORCE does not apply to them. This is exactly why the tests
        above need their own connection as a restricted role.
        """
        assert await _contents(db_session) == sorted([A_SECRET, B_SECRET])

    async def test_write_under_wrong_tenant_is_rejected(self, app_session, tenants):
        with tenant_scope(tenants[A_SLUG]):
            app_session.add(ContentChunk(
                model_id=tenants[B_SLUG], source_identifier="smuggled.md", source_url="",
                content="SMUGGLED", content_type="text", metadata_={}, embedding=[0.1] * 1024,
            ))
            with pytest.raises(Exception, match="row-level security"):
                await app_session.commit()


class TestGucLifecycle:
    """The setting is transaction-local, so when it gets applied is load-bearing."""

    async def test_tenant_bound_after_transaction_opens_sees_nothing(self, app_session, tenants):
        """Regression lock. Do not replace bind_tenant with a bare contextvar set.

        Resolving a slug to a model is itself a query, so the transaction is
        already open before the tenant is known and the after_begin listener has
        already run with nothing to publish. Setting only the contextvar at that
        point leaves the GUC empty for the rest of the request — every chunk read
        comes back empty and it looks like a retrieval bug, not a policy one.
        """
        await app_session.execute(text("SELECT 1"))  # autobegin, no tenant yet
        token = _MODEL_ID.set(tenants[A_SLUG])
        try:
            assert await _contents(app_session) == []
        finally:
            _MODEL_ID.reset(token)

    async def test_bind_tenant_repairs_an_open_transaction(self, app_session, tenants):
        await app_session.execute(text("SELECT 1"))
        await bind_tenant(app_session, tenants[A_SLUG])
        try:
            assert await _contents(app_session) == [A_SECRET]
        finally:
            _MODEL_ID.set(None)

    async def test_survives_a_mid_request_commit(self, app_session, tenants):
        """Request paths commit part-way through (see _log_message in api/chat.py).

        A commit discards the transaction-local setting; the listener re-applies
        it when the next transaction opens.
        """
        with tenant_scope(tenants[B_SLUG]):
            assert await _contents(app_session) == [B_SECRET]
            await app_session.commit()
            assert await _contents(app_session) == [B_SECRET]
