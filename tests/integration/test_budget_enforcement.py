"""Integration tests for monthly budget enforcement.

Covers the four scenarios that matter when the product goes public:

  1. Platform-keyed models are hard-capped at $10/month regardless of
     their configured budget_limit (so a viral post can't run up the bill).
  2. BYOK models bypass the platform cap and use their own budget_limit.
  3. BYOK models still get blocked once they exceed their own budget_limit.
  4. Token usage accumulates correctly across chat requests.

Budget state is seeded directly via the TokenUsage table — using mocked
token counts (~$0.0003 per call) it would take ~35k requests to organically
hit the cap, which is impractical in a test.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.rag_model import RagModel
from app.models.token_usage import TokenUsage


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


async def _seed_usage(db_session, model_id: int, cost: float) -> None:
    """Insert (or replace) the TokenUsage row for this model + current month."""
    db_session.add(TokenUsage(
        model_id=model_id,
        month=_current_month(),
        total_input_tokens=int(cost * 1_000_000),  # cosmetic, not used in check
        total_output_tokens=0,
        estimated_cost=cost,
    ))
    await db_session.commit()


async def _get_model(db_session, slug: str) -> RagModel:
    result = await db_session.execute(select(RagModel).where(RagModel.slug == slug))
    return result.scalar_one()


# ---------------------------------------------------------------------------
# Platform cap — applies to models with no BYOK keys
# ---------------------------------------------------------------------------


class TestPlatformCap:
    async def test_chat_blocked_when_platform_usage_exceeds_10(self, client, db_session):
        await client.post("/models", json={"name": "Capped", "slug": "capped-bot"})
        model = await _get_model(db_session, "capped-bot")
        await _seed_usage(db_session, model.id, cost=10.01)

        resp = await client.post("/models/capped-bot/chat", json={"question": "hi", "stream": False})
        assert resp.status_code == 429
        assert "budget" in resp.json()["detail"].lower()

    async def test_platform_cap_overrides_higher_configured_budget_limit(self, client, db_session):
        # User configures a $100 budget but uses platform keys — cap stays at $10.
        await client.post("/models", json={
            "name": "Greedy",
            "slug": "greedy-bot",
            "budget_limit": 100.0,
        })
        model = await _get_model(db_session, "greedy-bot")
        await _seed_usage(db_session, model.id, cost=10.01)

        resp = await client.post("/models/greedy-bot/chat", json={"question": "hi", "stream": False})
        assert resp.status_code == 429

    async def test_chat_allowed_just_under_platform_cap(self, client, db_session):
        await client.post("/models", json={"name": "Just Under", "slug": "just-under"})
        model = await _get_model(db_session, "just-under")
        await _seed_usage(db_session, model.id, cost=9.99)

        resp = await client.post("/models/just-under/chat", json={"question": "hi", "stream": False})
        # Anything other than 429 means the budget gate let it through.
        assert resp.status_code != 429


# ---------------------------------------------------------------------------
# BYOK — escapes the platform cap, uses its own budget_limit
# ---------------------------------------------------------------------------


class TestByokBudget:
    async def test_byok_escapes_platform_cap(self, client, db_session):
        # BYOK model with budget_limit well above $10, usage above platform cap.
        await client.post("/models", json={
            "name": "BYOK Big",
            "slug": "byok-big",
            "budget_limit": 100.0,
            "custom_anthropic_key": "sk-ant-test-fake",
            "custom_voyage_key": "pa-test-fake",
        })
        model = await _get_model(db_session, "byok-big")
        await _seed_usage(db_session, model.id, cost=50.0)

        resp = await client.post("/models/byok-big/chat", json={"question": "hi", "stream": False})
        # $50 usage on a $100 BYOK budget → not blocked by budget gate.
        assert resp.status_code != 429

    async def test_byok_blocked_when_own_budget_exceeded(self, client, db_session):
        await client.post("/models", json={
            "name": "BYOK Small",
            "slug": "byok-small",
            "budget_limit": 5.0,
            "custom_anthropic_key": "sk-ant-test-fake",
            "custom_voyage_key": "pa-test-fake",
        })
        model = await _get_model(db_session, "byok-small")
        await _seed_usage(db_session, model.id, cost=5.01)

        resp = await client.post("/models/byok-small/chat", json={"question": "hi", "stream": False})
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Accumulation — token usage actually increases after each chat
# ---------------------------------------------------------------------------


class TestUsageAccumulation:
    async def test_usage_row_created_and_incremented_after_chat(self, client, db_session):
        await client.post("/models", json={"name": "Accum", "slug": "accum-bot"})
        model = await _get_model(db_session, "accum-bot")

        # No usage row yet
        result = await db_session.execute(
            select(TokenUsage).where(
                TokenUsage.model_id == model.id,
                TokenUsage.month == _current_month(),
            )
        )
        assert result.scalar_one_or_none() is None

        resp = await client.post("/models/accum-bot/chat", json={"question": "hi", "stream": False})
        assert resp.status_code == 200

        # Usage row exists with the mocked 100 in / 50 out tokens
        result = await db_session.execute(
            select(TokenUsage).where(
                TokenUsage.model_id == model.id,
                TokenUsage.month == _current_month(),
            )
        )
        usage = result.scalar_one()
        assert usage.total_input_tokens == 100
        assert usage.total_output_tokens == 50
        assert usage.estimated_cost > 0

        # Second call — same row, accumulated counts
        resp2 = await client.post("/models/accum-bot/chat", json={"question": "again", "stream": False})
        assert resp2.status_code == 200

        await db_session.refresh(usage)
        assert usage.total_input_tokens == 200
        assert usage.total_output_tokens == 100
