"""Integration tests for model CRUD lifecycle.

Each test is self-contained — creates what it needs, no cross-test dependencies.
Savepoint rollback ensures cleanup.
"""

import pytest


class TestModelCreate:
    async def test_create_model(self, client):
        resp = await client.post("/models", json={
            "name": "Test Bot",
            "slug": "test-bot",
            "description": "A bot for integration testing",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Bot"
        assert data["slug"] == "test-bot"
        assert data["is_active"] is True
        assert data["has_custom_anthropic_key"] is False
        assert data["max_tokens"] == 1024

    async def test_create_and_get_model(self, client):
        await client.post("/models", json={"name": "Get Bot", "slug": "get-bot"})
        resp = await client.get("/models/get-bot")
        assert resp.status_code == 200
        assert resp.json()["slug"] == "get-bot"

    async def test_create_and_get_public_info(self, client):
        await client.post("/models", json={"name": "Public Bot", "slug": "public-bot"})
        resp = await client.get("/models/public-bot/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Public Bot"
        assert data["hosted_chat"] is True
        assert "system_prompt" not in data

    async def test_create_and_list_models(self, client):
        await client.post("/models", json={"name": "List Bot", "slug": "list-bot"})
        resp = await client.get("/models")
        assert resp.status_code == 200
        slugs = [m["slug"] for m in resp.json()]
        assert "list-bot" in slugs

    async def test_duplicate_slug_rejected(self, client):
        await client.post("/models", json={"name": "First", "slug": "dup-bot"})
        resp = await client.post("/models", json={"name": "Second", "slug": "dup-bot"})
        assert resp.status_code == 409


class TestModelUpdate:
    async def test_update_description(self, client):
        await client.post("/models", json={"name": "Update Bot", "slug": "update-bot"})
        resp = await client.patch("/models/update-bot", json={
            "description": "Updated",
            "similarity_threshold": 0.5,
        })
        assert resp.status_code == 200
        assert resp.json()["description"] == "Updated"
        assert resp.json()["similarity_threshold"] == 0.5

    async def test_update_name(self, client):
        await client.post("/models", json={"name": "Rename Bot", "slug": "rename-bot"})
        resp = await client.patch("/models/rename-bot", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed"


class TestModelDelete:
    async def test_delete_model(self, client):
        await client.post("/models", json={"name": "Delete Bot", "slug": "delete-bot"})
        resp = await client.delete("/models/delete-bot")
        assert resp.status_code == 204

        # Should be gone
        resp = await client.get("/models/delete-bot")
        assert resp.status_code == 404


class TestModelNotFound:
    async def test_get_nonexistent_model(self, client):
        resp = await client.get("/models/nonexistent")
        assert resp.status_code == 404


class TestModelValidation:
    async def test_invalid_slug_uppercase(self, client):
        resp = await client.post("/models", json={"name": "Bad", "slug": "Bad-Slug"})
        assert resp.status_code == 422

    async def test_invalid_generation_model(self, client):
        resp = await client.post("/models", json={
            "name": "Bad", "slug": "bad-model", "generation_model": "gpt-4",
        })
        assert resp.status_code == 422

    async def test_chunk_size_out_of_range(self, client):
        resp = await client.post("/models", json={
            "name": "Bad", "slug": "bad-chunks", "chunk_size": 50,
        })
        assert resp.status_code == 422
