"""Integration tests for cross-tenant isolation.

Seeds a second tenant (different owner_id) directly in the DB, then verifies
that a request authenticated as the test user cannot read or modify any of
the other tenant's resources, and that hybrid retrieval is strictly scoped
by model_id.
"""

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.models.content import ContentChunk
from app.models.ingestion_source import IngestionSource
from app.models.model_api_key import ModelApiKey
from app.models.rag_model import RagModel
from app.models.user import User
from app.services.retrieval import _keyword_search, _vector_search

OTHER_USER_ID = "other-tenant-user"
OTHER_SLUG = "other-tenant-bot"
OTHER_SOURCE_ID = "other-secret-doc"
OTHER_SECRET = "OTHER_TENANT_SECRET_PHRASE_THAT_SHOULD_NEVER_LEAK"


@pytest_asyncio.fixture
async def other_tenant(db_session):
    """Create a second tenant owned by OTHER_USER_ID with seeded content + API key."""
    db_session.add(User(
        clerk_user_id=OTHER_USER_ID,
        email="other@example.com",
        first_name="Other",
        last_name="User",
        allow_global_keys=True,
    ))

    model = RagModel(
        owner_id=OTHER_USER_ID,
        name="Other Tenant",
        slug=OTHER_SLUG,
        embedding_model="voyage-4-lite",
        generation_model="claude-haiku-4-5",
    )
    db_session.add(model)
    await db_session.flush()

    db_session.add(IngestionSource(
        model_id=model.id,
        source_identifier=OTHER_SOURCE_ID,
        content_hash="other-hash",
        chunk_count=1,
        source_url=f"text://{OTHER_SOURCE_ID}",
        content_type="text",
        status="complete",
        raw_content=OTHER_SECRET,
    ))
    db_session.add(ContentChunk(
        model_id=model.id,
        content=OTHER_SECRET,
        embedding=[0.1] * 1024,
        source_url=f"text://{OTHER_SOURCE_ID}",
        source_identifier=OTHER_SOURCE_ID,
        content_type="text",
    ))
    db_session.add(ModelApiKey(
        model_id=model.id,
        label="other-tenant-key",
        key_hash="$2b$12$" + "x" * 53,  # well-formed but unmatchable bcrypt hash
        key_prefix="ragr_otherx",
    ))
    await db_session.commit()

    # Re-fetch to detach from the flush state and return a stable handle
    result = await db_session.execute(select(RagModel).where(RagModel.slug == OTHER_SLUG))
    return result.scalar_one()


# ---------------------------------------------------------------------------
# Listing must scope by owner
# ---------------------------------------------------------------------------


class TestListScoping:
    async def test_list_models_excludes_other_tenants(self, client, other_tenant):
        # Create one model owned by the test user so we can confirm presence vs. absence.
        await client.post("/models", json={"name": "Mine", "slug": "mine-bot"})

        resp = await client.get("/models")
        assert resp.status_code == 200
        slugs = {m["slug"] for m in resp.json()}
        assert "mine-bot" in slugs
        assert OTHER_SLUG not in slugs, (
            "Cross-tenant leak: list_models returned another user's model"
        )


# ---------------------------------------------------------------------------
# Authenticated tenant-scoped routes must reject cross-tenant access
# ---------------------------------------------------------------------------


# (method, path_template, json_body) — path_template uses {slug}
_CROSS_TENANT_ROUTES = [
    # models.py
    ("GET", "/models/{slug}", None),
    ("PATCH", "/models/{slug}", {"description": "pwned"}),
    ("DELETE", "/models/{slug}", None),
    ("PATCH", "/models/{slug}/theme", {"label": "pwned"}),

    # sources.py
    ("GET", "/models/{slug}/sources", None),
    ("DELETE", "/models/{slug}/sources", None),
    ("POST", "/models/{slug}/sources", {"content": "x", "source_identifier": "evil"}),

    # api_keys.py
    ("GET", "/models/{slug}/api-keys", None),
    ("POST", "/models/{slug}/api-keys", {"label": "evil"}),

    # admin.py
    ("GET", "/models/{slug}/stats", None),
    ("GET", "/models/{slug}/stats/daily", None),
    ("GET", "/models/{slug}/top-sources", None),
    ("GET", "/models/{slug}/conversations", None),
    ("GET", "/models/{slug}/system-prompt-history", None),
]


@pytest.mark.parametrize("method,path,body", _CROSS_TENANT_ROUTES)
async def test_authenticated_route_rejects_cross_tenant(
        client, other_tenant, method, path, body,
):
    """Every authenticated tenant-scoped route returns 403 (or 404) when accessing another tenant."""
    url = path.format(slug=OTHER_SLUG)
    resp = await client.request(method, url, json=body)
    assert resp.status_code in (403, 404), (
        f"{method} {url} returned {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# Cross-tenant access to nested resources by ID must also fail
# ---------------------------------------------------------------------------


class TestNestedResourceIsolation:
    async def test_cannot_read_other_tenants_source_by_id(self, client, other_tenant, db_session):
        result = await db_session.execute(
            select(IngestionSource).where(IngestionSource.model_id == other_tenant.id)
        )
        source = result.scalar_one()

        resp = await client.get(f"/models/{OTHER_SLUG}/sources/{source.id}")
        assert resp.status_code in (403, 404)

    async def test_cannot_delete_other_tenants_source_by_id(self, client, other_tenant, db_session):
        result = await db_session.execute(
            select(IngestionSource).where(IngestionSource.model_id == other_tenant.id)
        )
        source = result.scalar_one()

        resp = await client.delete(f"/models/{OTHER_SLUG}/sources/{source.id}")
        assert resp.status_code in (403, 404)

        # Source must still exist
        result = await db_session.execute(
            select(IngestionSource).where(IngestionSource.id == source.id)
        )
        assert result.scalar_one_or_none() is not None

    async def test_cannot_revoke_other_tenants_api_key(self, client, other_tenant, db_session):
        result = await db_session.execute(
            select(ModelApiKey).where(ModelApiKey.model_id == other_tenant.id)
        )
        api_key = result.scalar_one()

        resp = await client.delete(f"/models/{OTHER_SLUG}/api-keys/{api_key.id}")
        assert resp.status_code in (403, 404)


# ---------------------------------------------------------------------------
# Public endpoints — sanity check the slug is reachable, so the 403s above
# are real ownership rejections rather than 404s from a missing model.
# ---------------------------------------------------------------------------


class TestPublicEndpointsReachable:
    async def test_public_info_works_for_other_tenant(self, client, other_tenant):
        resp = await client.get(f"/models/{OTHER_SLUG}/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == OTHER_SLUG
        # Public schema must not include keys, system_prompt, or owner info
        assert "system_prompt" not in data
        assert "custom_anthropic_key" not in data
        assert "owner_id" not in data

    async def test_public_theme_works_for_other_tenant(self, client, other_tenant):
        resp = await client.get(f"/models/{OTHER_SLUG}/theme")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Retrieval is strictly scoped by model_id — even when both tenants have
# similar content, vector + keyword search must only return the queried
# tenant's chunks.
# ---------------------------------------------------------------------------


class TestRetrievalScoping:
    async def test_vector_search_only_returns_queried_models_chunks(
            self, client, other_tenant, db_session,
    ):
        # Create a model + chunk for the test user with a distinct phrase
        await client.post("/models", json={"name": "Mine", "slug": "mine-retrieval"})
        result = await db_session.execute(select(RagModel).where(RagModel.slug == "mine-retrieval"))
        my_model = result.scalar_one()

        my_secret = "MY_TENANT_DISTINCT_PHRASE_FOR_RETRIEVAL_TEST"
        db_session.add(ContentChunk(
            model_id=my_model.id,
            content=my_secret,
            embedding=[0.1] * 1024,  # same vector as the other tenant's chunk
            source_url="text://mine",
            source_identifier="mine-doc",
            content_type="text",
        ))
        await db_session.commit()

        # Identical query embedding — both chunks would match if scoping were broken
        query_embedding = [0.1] * 1024

        my_hits = await _vector_search(db_session, my_model, query_embedding, threshold_distance=2.0, limit=10)
        other_hits = await _vector_search(db_session, other_tenant, query_embedding, threshold_distance=2.0, limit=10)

        my_contents = {c.content for c, _ in my_hits}
        other_contents = {c.content for c, _ in other_hits}

        assert my_secret in my_contents
        assert OTHER_SECRET not in my_contents, "Vector search leaked another tenant's chunk"

        assert OTHER_SECRET in other_contents
        assert my_secret not in other_contents, "Vector search leaked another tenant's chunk"

    async def test_keyword_search_only_returns_queried_models_chunks(
            self, client, other_tenant, db_session,
    ):
        await client.post("/models", json={"name": "Mine KW", "slug": "mine-keyword"})
        result = await db_session.execute(select(RagModel).where(RagModel.slug == "mine-keyword"))
        my_model = result.scalar_one()

        # Reuse the same secret phrase the other tenant has, to prove scoping
        # rather than content uniqueness is what isolates the results.
        db_session.add(ContentChunk(
            model_id=my_model.id,
            content=OTHER_SECRET + " (mine)",
            embedding=[0.0] * 1024,
            source_url="text://mine-kw",
            source_identifier="mine-kw-doc",
            content_type="text",
        ))
        await db_session.commit()

        # Refresh search_vector — populated by trigger; flush via raw SQL
        from sqlalchemy import text
        await db_session.execute(text(
            "UPDATE content_chunks SET search_vector = to_tsvector('english', content) "
            "WHERE search_vector IS NULL"
        ))
        await db_session.commit()

        my_hits = await _keyword_search(db_session, my_model, "OTHER_TENANT_SECRET_PHRASE", limit=10)
        other_hits = await _keyword_search(db_session, other_tenant, "OTHER_TENANT_SECRET_PHRASE", limit=10)

        my_chunk_model_ids = {c.model_id for c, _ in my_hits}
        other_chunk_model_ids = {c.model_id for c, _ in other_hits}

        assert my_chunk_model_ids <= {my_model.id}, "Keyword search leaked another tenant's chunk"
        assert other_chunk_model_ids <= {other_tenant.id}, "Keyword search leaked another tenant's chunk"
