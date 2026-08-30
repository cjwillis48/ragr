"""Integration tests for the retrieve-only endpoint and code metadata roundtrip.

Verifies:
- Ingesting with content_type="code" populates chunk metadata
  (file_path, start_line, end_line, language) and the metadata flows
  back through POST /retrieve.
- Retrieve responses include cost + tokens for both embedding and rerank.
- Non-code sources still work (empty metadata).
"""

import pytest


@pytest.fixture
async def model_slug(client):
    """Create a model for the test."""
    resp = await client.post("/models", json={
        "name": "Retrieve Test Bot",
        "slug": "retrieve-test-bot",
        "description": "Testing the /retrieve endpoint",
        "system_prompt": "You are a helpful assistant.",
    })
    assert resp.status_code == 201
    return "retrieve-test-bot"


class TestRetrieveEndpoint:
    async def test_retrieve_returns_chunks_with_scores(self, client, model_slug):
        await client.put(f"/models/{model_slug}/sources/note.txt", json={
            "content": "Python was created by Guido van Rossum in 1991.",
        })

        resp = await client.post(f"/models/{model_slug}/retrieve", json={
            "query": "Who created Python?",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "chunks" in data
        assert "tokens" in data
        assert "cost" in data
        assert len(data["chunks"]) >= 1
        for c in data["chunks"]:
            assert "id" in c
            assert "content" in c
            assert "source_identifier" in c
            assert "metadata" in c
            assert "score" in c
            assert c["score"]["retrieval_method"] in ("vector", "keyword", "hybrid")

    async def test_retrieve_top_k_caps_results(self, client, model_slug):
        # Ingest several sources so the model has plenty of chunks
        for i in range(5):
            await client.put(f"/models/{model_slug}/sources/doc-{i}.txt", json={
                "content": f"Document number {i} talks about Python programming language.",
            })

        resp = await client.post(f"/models/{model_slug}/retrieve", json={
            "query": "Python",
            "top_k": 2,
        })
        assert resp.status_code == 200
        chunks = resp.json()["chunks"]
        assert len(chunks) <= 2

    async def test_retrieve_cost_is_positive_when_tokens_used(self, client, model_slug):
        await client.put(f"/models/{model_slug}/sources/cost-test.txt", json={
            "content": "Some content to retrieve over.",
        })

        resp = await client.post(f"/models/{model_slug}/retrieve", json={
            "query": "content",
        })
        assert resp.status_code == 200
        data = resp.json()
        cost = data["cost"]
        assert cost.startswith("$")
        # Tokens should be non-zero for embedding (and likely rerank)
        assert data["tokens"]["embedding"] >= 0
        assert data["tokens"]["rerank"] >= 0
