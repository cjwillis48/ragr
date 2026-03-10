from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag_model import RagModel
from app.models.token_usage import TokenUsage

# Approximate pricing per 1M tokens (USD)
# These are rough estimates — update as pricing changes
MODEL_PRICING = {
    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
}
DEFAULT_PRICING = {"input": 3.0, "output": 15.0}

# Voyage embedding pricing per 1M tokens (USD)
EMBEDDING_PRICING = {
    "voyage-4-lite": 0.02,
    "voyage-3-lite": 0.02,
    "voyage-3": 0.06,
    "voyage-code-3": 0.18,
}
DEFAULT_EMBEDDING_PRICING = 0.02

# Voyage reranker pricing per 1M tokens (USD)
RERANK_PRICING = {
    "rerank-2": 0.05,
    "rerank-2-lite": 0.02,
    "rerank-2.5-lite": 0.02,
}
DEFAULT_RERANK_PRICING = 0.05


def estimate_embedding_cost(model_name: str, total_tokens: int) -> float:
    """Estimate embedding cost in USD for a given number of tokens."""
    price_per_m = EMBEDDING_PRICING.get(model_name, DEFAULT_EMBEDDING_PRICING)
    return total_tokens * price_per_m / 1_000_000


def estimate_rerank_cost(model_name: str, total_tokens: int) -> float:
    """Estimate reranking cost in USD for a given number of tokens."""
    price_per_m = RERANK_PRICING.get(model_name, DEFAULT_RERANK_PRICING)
    return total_tokens * price_per_m / 1_000_000


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given number of tokens."""
    pricing = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


async def get_current_month_usage(
    session: AsyncSession,
    model: RagModel,
) -> TokenUsage | None:
    """Get the current month's token usage for a model."""
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    result = await session.execute(
        select(TokenUsage).where(
            TokenUsage.model_id == model.id,
            TokenUsage.month == month,
        )
    )
    return result.scalar_one_or_none()


async def record_usage(
    session: AsyncSession,
    model: RagModel,
    input_tokens: int,
    output_tokens: int,
) -> TokenUsage:
    """Record token usage for the current month."""
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    cost = estimate_cost(model.generation_model, input_tokens, output_tokens)

    usage = await get_current_month_usage(session, model)
    if usage is None:
        usage = TokenUsage(
            model_id=model.id,
            month=month,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            estimated_cost=cost,
        )
        session.add(usage)
    else:
        usage.total_input_tokens += input_tokens
        usage.total_output_tokens += output_tokens
        usage.estimated_cost += cost

    await session.flush()
    return usage


async def check_budget(session: AsyncSession, model: RagModel) -> bool:
    """Check if a model is within budget. Returns True if OK to proceed."""
    usage = await get_current_month_usage(session, model)
    if usage is None:
        return True
    return usage.estimated_cost < model.budget_limit
