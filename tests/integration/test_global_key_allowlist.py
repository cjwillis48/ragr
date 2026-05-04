"""Integration tests for the global API key allowlist.

By default, users cannot create or use models that rely on the platform's
Anthropic + Voyage keys. The `users.allow_global_keys` flag opts an account
in. Tests start with the test user allowlisted (via the `auth_headers`
fixture) and use `set_test_user_allowlist` to flip the flag mid-test.
"""

import pytest

from sqlalchemy import select

from app.models.user import User


class TestModelCreateGate:
    async def test_create_with_no_keys_fails_when_not_allowlisted(self, client, set_test_user_allowlist):
        await set_test_user_allowlist(False)
        resp = await client.post("/models", json={"name": "Blocked", "slug": "blocked-bot"})
        assert resp.status_code == 403
        assert "platform" in resp.json()["detail"].lower()

    async def test_create_with_no_keys_succeeds_when_allowlisted(self, client):
        # auth_headers seeds allow_global_keys=True
        resp = await client.post("/models", json={"name": "Allowed", "slug": "allowed-bot"})
        assert resp.status_code == 201

    async def test_create_with_byok_succeeds_for_non_allowlisted(self, client, set_test_user_allowlist):
        await set_test_user_allowlist(False)
        resp = await client.post("/models", json={
            "name": "BYOK Bot",
            "slug": "byok-bot",
            "custom_anthropic_key": "sk-ant-test-fake-key",
            "custom_voyage_key": "pa-test-fake-voyage-key",
        })
        assert resp.status_code == 201

    async def test_create_with_only_anthropic_key_fails_for_non_allowlisted(self, client, set_test_user_allowlist):
        await set_test_user_allowlist(False)
        resp = await client.post("/models", json={
            "name": "Half BYOK",
            "slug": "half-byok",
            "custom_anthropic_key": "sk-ant-test-fake-key",
        })
        assert resp.status_code == 403


class TestModelUpdateGate:
    async def test_clearing_keys_fails_for_non_allowlisted(self, client, set_test_user_allowlist):
        # Create with BYOK while allowlisted to make the row exist
        resp = await client.post("/models", json={
            "name": "Will Lose Keys",
            "slug": "lose-keys",
            "custom_anthropic_key": "sk-ant-x",
            "custom_voyage_key": "pa-x",
        })
        assert resp.status_code == 201

        await set_test_user_allowlist(False)
        resp = await client.patch("/models/lose-keys", json={"custom_anthropic_key": None})
        assert resp.status_code == 403

    async def test_setting_byok_keeps_working_for_non_allowlisted(self, client, set_test_user_allowlist):
        resp = await client.post("/models", json={
            "name": "Keeps Keys",
            "slug": "keeps-keys",
            "custom_anthropic_key": "sk-ant-x",
            "custom_voyage_key": "pa-x",
        })
        assert resp.status_code == 201

        await set_test_user_allowlist(False)
        resp = await client.patch("/models/keeps-keys", json={"description": "still has BYOK"})
        assert resp.status_code == 200


class TestChatGate:
    async def test_chat_blocked_when_owner_not_allowlisted_and_no_byok(self, client, set_test_user_allowlist):
        # Create a NULL-key model while allowlisted, then revoke
        resp = await client.post("/models", json={"name": "Will Be Blocked", "slug": "will-block"})
        assert resp.status_code == 201

        await set_test_user_allowlist(False)
        resp = await client.post("/models/will-block/chat", json={"message": "hello"})
        assert resp.status_code == 403
        assert "platform" in resp.json()["detail"].lower()

    async def test_chat_allowed_when_owner_allowlisted(self, client):
        resp = await client.post("/models", json={"name": "Chat OK", "slug": "chat-ok"})
        assert resp.status_code == 201

        # We only care that the allowlist gate doesn't fire — full chat plumbing
        # is covered by tests/integration/test_ingest_and_chat.py.
        resp = await client.post("/models/chat-ok/chat", json={"message": "hi", "stream": False})
        assert resp.status_code != 403

    async def test_chat_allowed_with_byok_even_when_not_allowlisted(self, client, set_test_user_allowlist):
        resp = await client.post("/models", json={
            "name": "BYOK Chat",
            "slug": "byok-chat",
            "custom_anthropic_key": "sk-ant-x",
            "custom_voyage_key": "pa-x",
        })
        assert resp.status_code == 201

        await set_test_user_allowlist(False)
        resp = await client.post("/models/byok-chat/chat", json={"message": "hi", "stream": False})
        assert resp.status_code != 403


class TestUsersTableSync:
    async def test_authenticated_request_creates_user_row(self, client, db_session, test_user_id):
        # Make any authenticated request — auth_headers fixture already seeded the row
        resp = await client.get("/models")
        assert resp.status_code == 200

        result = await db_session.execute(select(User).where(User.clerk_user_id == test_user_id))
        user = result.scalar_one_or_none()
        assert user is not None
        assert user.allow_global_keys is True


class TestCurrentUserEndpoint:
    async def test_me_returns_allowlisted(self, client, test_user_id):
        resp = await client.get("/users/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == test_user_id
        assert data["allow_global_keys"] is True

    async def test_me_reflects_revoked_allowlist(self, client, set_test_user_allowlist):
        await set_test_user_allowlist(False)
        resp = await client.get("/users/me")
        assert resp.status_code == 200
        assert resp.json()["allow_global_keys"] is False
