import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.budget import (
    estimate_cost,
    estimate_embedding_cost,
    estimate_rerank_cost,
    check_budget,
    MODEL_PRICING,
    PLATFORM_BUDGET_CAP,
)
from app.schemas.models import SUPPORTED_GENERATION_MODELS


class TestEstimateCost:
    def test_known_model_haiku(self):
        # haiku: input=0.80, output=4.0 per 1M
        cost = estimate_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
        assert cost == pytest.approx(4.80)

    def test_known_model_sonnet(self):
        # sonnet: input=3.0, output=15.0 per 1M
        cost = estimate_cost("claude-sonnet-4-6", 1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_known_model_opus(self):
        # opus: input=15.0, output=75.0 per 1M
        cost = estimate_cost("claude-opus-4-6", 1_000_000, 1_000_000)
        assert cost == pytest.approx(90.0)

    def test_unknown_model_uses_default(self):
        # DEFAULT: input=3.0, output=15.0
        cost = estimate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == pytest.approx(18.0)

    def test_zero_tokens(self):
        assert estimate_cost("claude-haiku-4-5", 0, 0) == 0.0

    def test_small_token_count(self):
        cost = estimate_cost("claude-haiku-4-5", 1000, 500)
        assert cost == pytest.approx((1000 * 0.80 + 500 * 4.0) / 1_000_000)


class TestModelPricingSync:
    def test_every_supported_model_has_pricing(self):
        """Guard against pricing drift — every model in the allowlist must have a pricing entry."""
        missing = SUPPORTED_GENERATION_MODELS - MODEL_PRICING.keys()
        assert not missing, f"Models missing from MODEL_PRICING: {missing}"

    def test_pricing_has_no_stale_models(self):
        """Pricing dict should not contain models that are no longer supported."""
        stale = MODEL_PRICING.keys() - SUPPORTED_GENERATION_MODELS
        assert not stale, f"Stale models in MODEL_PRICING: {stale}"


class TestEstimateEmbeddingCost:
    def test_known_model(self):
        cost = estimate_embedding_cost("voyage-4-lite", 1_000_000)
        assert cost == pytest.approx(0.02)

    def test_unknown_model_default(self):
        cost = estimate_embedding_cost("unknown", 1_000_000)
        assert cost == pytest.approx(0.02)

    def test_zero_tokens(self):
        assert estimate_embedding_cost("voyage-4-lite", 0) == 0.0


class TestEstimateRerankCost:
    def test_known_model(self):
        cost = estimate_rerank_cost("rerank-2.5-lite", 1_000_000)
        assert cost == pytest.approx(0.02)

    def test_unknown_model_default(self):
        cost = estimate_rerank_cost("unknown", 1_000_000)
        assert cost == pytest.approx(0.05)


class TestCheckBudget:
    async def test_no_usage_returns_true(self, sample_model):
        session = AsyncMock()
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=result)

        assert await check_budget(session, sample_model) is True

    async def test_under_limit_returns_true(self, sample_model):
        session = AsyncMock()
        usage = MagicMock()
        usage.estimated_cost = 5.0
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=usage)
        session.execute = AsyncMock(return_value=result)
        sample_model.budget_limit = 10.0
        sample_model.custom_anthropic_key = None

        assert await check_budget(session, sample_model) is True

    async def test_over_limit_returns_false(self, sample_model):
        session = AsyncMock()
        usage = MagicMock()
        usage.estimated_cost = 11.0
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=usage)
        session.execute = AsyncMock(return_value=result)
        sample_model.budget_limit = 10.0
        sample_model.custom_anthropic_key = None

        assert await check_budget(session, sample_model) is False

    async def test_platform_key_hard_cap(self, sample_model):
        """Platform keys are capped at $10 even if budget_limit is higher."""
        session = AsyncMock()
        usage = MagicMock()
        usage.estimated_cost = 9.5
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=usage)
        session.execute = AsyncMock(return_value=result)
        sample_model.budget_limit = 50.0  # Higher than cap
        sample_model.custom_anthropic_key = None

        assert await check_budget(session, sample_model) is True

        usage.estimated_cost = 10.5
        assert await check_budget(session, sample_model) is False

    async def test_custom_key_respects_budget_limit(self, sample_model):
        """Custom key models use their own budget_limit, not the platform cap."""
        session = AsyncMock()
        usage = MagicMock()
        usage.estimated_cost = 15.0
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=usage)
        session.execute = AsyncMock(return_value=result)
        sample_model.budget_limit = 50.0
        sample_model.custom_anthropic_key = "sk-custom-key"

        # 15 < 50, so should be True
        assert await check_budget(session, sample_model) is True

        usage.estimated_cost = 55.0
        assert await check_budget(session, sample_model) is False
