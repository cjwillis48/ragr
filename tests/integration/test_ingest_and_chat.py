"""Integration tests for the ingest → chat pipeline.

Creates a model, ingests text content, then queries it via chat.
Anthropic is mocked (always), but DB operations are real.
"""

import pytest

import app.api.chat as chat_module
from httpx import ASGITransport, AsyncClient as HC


@pytest.fixture
async def model_slug(client):
    """Create a model for the test and return its slug. Cleaned up by savepoint rollback."""
    resp = await client.post("/models", json={
        "name": "Chat Test Bot",
        "slug": "chat-test-bot",
        "description": "For testing ingest and chat",
        "system_prompt": "You are a helpful assistant that answers questions about Python.",
    })
    assert resp.status_code == 201
    return "chat-test-bot"


class TestTextIngestion:
    async def test_ingest_text_content(self, client, model_slug):
        resp = await client.post(f"/models/{model_slug}/sources", json={
            "source_identifier": "python-basics",
            "content": (
                "Python is a high-level programming language. "
                "It supports multiple paradigms including procedural, "
                "object-oriented, and functional programming. "
                "Python uses dynamic typing and garbage collection. "
                "It was created by Guido van Rossum and first released in 1991."
            ),
            "content_type": "text",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "complete"
        assert data[0]["chunks_created"] >= 1

    async def test_ingest_is_idempotent(self, client, model_slug):
        content = "This is idempotent test content that should only be ingested once."

        # First ingest
        resp1 = await client.post(f"/models/{model_slug}/sources", json={
            "source_identifier": "idempotent-test",
            "content": content,
        })
        assert resp1.status_code == 200
        assert resp1.json()[0]["skipped"] is False

        # Second ingest — same content, should be skipped
        resp2 = await client.post(f"/models/{model_slug}/sources", json={
            "source_identifier": "idempotent-test",
            "content": content,
        })
        assert resp2.status_code == 200
        assert resp2.json()[0]["skipped"] is True

    async def test_list_sources(self, client, model_slug):
        # Ingest something first
        await client.post(f"/models/{model_slug}/sources", json={
            "source_identifier": "list-test",
            "content": "Content for listing sources.",
        })

        resp = await client.get(f"/models/{model_slug}/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        identifiers = [s["source_identifier"] for s in data["sources"]]
        assert "list-test" in identifiers

    async def test_delete_source(self, client, model_slug):
        # Ingest
        await client.post(f"/models/{model_slug}/sources", json={
            "source_identifier": "delete-me",
            "content": "Content to be deleted.",
        })

        # Find the source ID
        sources_resp = await client.get(f"/models/{model_slug}/sources")
        source = next(s for s in sources_resp.json()["sources"] if s["source_identifier"] == "delete-me")

        # Delete
        resp = await client.delete(f"/models/{model_slug}/sources/{source['id']}")
        assert resp.status_code == 204


class TestChat:
    """Test the chat endpoint with ingested content. Anthropic is mocked."""

    @pytest.fixture(autouse=True)
    async def _ingest_content(self, client, model_slug):
        """Ensure content is ingested before chat tests."""
        await client.post(f"/models/{model_slug}/sources", json={
            "source_identifier": "chat-knowledge",
            "content": (
                "Python was created by Guido van Rossum. "
                "The first version was released in 1991. "
                "Python 3.0 was released in 2008 with many improvements."
            ),
        })

    async def test_chat_non_streaming(self, client, model_slug):
        resp = await client.post(f"/models/{model_slug}/chat", json={
            "question": "Who created Python?",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["status"] in ("answered", "unanswered", "off_topic")
        assert "session_id" in data
        assert data["tokens_in"] >= 0
        assert data["tokens_out"] >= 0

    async def test_chat_with_session_id(self, client, model_slug):
        session_id = "test-session-123"

        resp1 = await client.post(f"/models/{model_slug}/chat", json={
            "question": "What is Python?",
            "session_id": session_id,
        })
        assert resp1.status_code == 200
        assert resp1.json()["session_id"] == session_id

        # Second message in same session
        resp2 = await client.post(f"/models/{model_slug}/chat", json={
            "question": "When was it created?",
            "session_id": session_id,
        })
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id

    async def test_chat_streaming(self, client, model_slug, monkeypatch):
        # Stub _log_message for streaming — it opens its own async_session
        # which can't see the test transaction's uncommitted model row.
        async def _noop_log(*args, **kwargs):
            pass

        monkeypatch.setattr(chat_module, "_log_message", _noop_log)

        resp = await client.post(
            f"/models/{model_slug}/chat",
            json={"question": "Tell me about Python", "stream": True},
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/event-stream")

        body = resp.text
        assert "event: done" in body

    async def test_chat_returns_cost(self, client, model_slug):
        resp = await client.post(f"/models/{model_slug}/chat", json={
            "question": "What is Python?",
        })
        assert resp.status_code == 200
        cost = resp.json().get("cost")
        assert cost is not None
        assert cost.startswith("$")

    async def test_chat_no_auth_hosted_model(self, client, model_slug, app):
        """Hosted chat models allow unauthenticated access."""
        # Make a request without auth headers
        transport = ASGITransport(app=app)
        async with HC(transport=transport, base_url="http://test") as unauth_client:
            resp = await unauth_client.post(f"/models/{model_slug}/chat", json={
                "question": "Hello",
            })
            # Should work because hosted_chat defaults to True
            assert resp.status_code == 200


class TestStats:
    """Test stats endpoints after ingestion + chat."""

    @pytest.fixture(autouse=True)
    async def _setup(self, client, model_slug):
        """Ingest content and send a chat message."""
        await client.post(f"/models/{model_slug}/sources", json={
            "source_identifier": "stats-content",
            "content": "Knowledge for stats testing. Python is great.",
        })
        await client.post(f"/models/{model_slug}/chat", json={
            "question": "What is Python?",
        })

    async def test_model_stats(self, client, model_slug):
        resp = await client.get(f"/models/{model_slug}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] >= 1
        assert data["total_messages"] >= 1
        assert data["total_sources"] >= 1

    async def test_conversations_listed(self, client, model_slug):
        resp = await client.get(f"/models/{model_slug}/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["conversations"]) >= 1
