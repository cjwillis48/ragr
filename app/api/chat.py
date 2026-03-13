import json
import logging
import time
import uuid

import anthropic
import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.dependencies import require_chat_auth
from app.models.conversation import ConversationLog
from app.models.rag_model import RagModel
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.budget import check_budget, estimate_cost, estimate_rerank_cost, record_usage
from app.services.generation import GenerationResult, generate_answer, generate_answer_stream
from app.services.retrieval import RetrievalResult, retrieve_with_threshold

router = APIRouter(tags=["chat"])
logger = logging.getLogger("ragr.chat")


async def _load_session_history(
    session: AsyncSession,
    model: RagModel,
    session_id: str,
) -> list[dict]:
    """Load the last N conversation turns for a session from the DB."""
    result = await session.execute(
        select(ConversationLog)
        .where(
            ConversationLog.model_id == model.id,
            ConversationLog.session_id == session_id,
        )
        .order_by(ConversationLog.created_at.desc())
        .limit(model.history_turns)
    )
    rows = result.scalars().all()

    # Reverse to chronological order and flatten to message pairs
    history = []
    for row in reversed(rows):
        history.append({"role": "user", "content": row.question})
        history.append({"role": "assistant", "content": row.answer})
    return history


async def _log_conversation(
    session: AsyncSession,
    model: RagModel,
    question: str,
    answer: str,
    status: str,
    tokens_in: int,
    tokens_out: int,
    session_id: str | None = None,
) -> None:
    """Record token usage and log the conversation."""
    await record_usage(session, model, tokens_in, tokens_out)
    session.add(ConversationLog(
        model_id=model.id,
        session_id=session_id,
        question=question,
        answer=answer,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    ))
    await session.commit()


@router.post("/models/{slug}/chat")
async def chat(
    body: ChatRequest,
    model: RagModel = Depends(require_chat_auth),
    session: AsyncSession = Depends(get_session),
):
    """Query a model — public endpoint. Set stream: true for SSE."""
    if not await check_budget(session, model):
        raise HTTPException(status_code=429, detail="Model has exceeded its monthly budget")

    # Resolve session ID
    session_id = body.session_id or str(uuid.uuid4())

    t_req = time.perf_counter()
    try:
        retrieval = await retrieve_with_threshold(session, model, body.question)
    except httpx.TimeoutException:
        logger.error("Voyage embedding/rerank timed out for model_id=%s", model.id)
        raise HTTPException(status_code=503, detail="Embedding service timed out. Please try again.")
    except Exception:
        logger.exception("Embedding/rerank failed for model_id=%s", model.id)
        raise HTTPException(status_code=503, detail="Embedding service unavailable. Please try again.")

    rerank_cost = estimate_rerank_cost(model.rerank_model, retrieval.rerank_tokens) if retrieval.rerank_tokens else 0.0
    logger.info("pre_stream_ready %.0fms chunks=%d rerank_cost=$%.6f", (time.perf_counter() - t_req) * 1000, len(retrieval.chunks), rerank_cost)

    # Build history: prefer server-side session history, fall back to client-provided
    if body.session_id or not body.history:
        history = await _load_session_history(session, model, session_id)
    else:
        history = [{"role": m.role, "content": m.content} for m in body.history]

    history = history or None

    if body.stream:
        return StreamingResponse(
            _stream_response(session, model, body.question, retrieval.chunks, history, session_id, rerank_cost),
            media_type="text/event-stream",
        )

    try:
        result = await generate_answer(model, body.question, retrieval.chunks, history=history)
    except anthropic.APIStatusError as e:
        if e.status_code == 529:
            raise HTTPException(status_code=503, detail="AI provider is temporarily overloaded. Please try again.")
        raise
    await _log_conversation(session, model, body.question, result.answer, result.status, result.input_tokens, result.output_tokens, session_id)

    generation_cost = estimate_cost(model.generation_model, result.input_tokens, result.output_tokens)

    return ChatResponse(
        answer=result.answer,
        status=result.status,
        session_id=session_id,
        tokens_in=result.input_tokens,
        tokens_out=result.output_tokens,
        cost=f"${generation_cost + rerank_cost:.6f}",
    )


async def _stream_response(
    session: AsyncSession,
    model: RagModel,
    question: str,
    chunks: list,
    history: list[dict] | None = None,
    session_id: str | None = None,
    rerank_cost: float = 0.0,
):
    """SSE generator. Streams text deltas, then a final done event with metadata."""
    try:
        async for event in generate_answer_stream(model, question, chunks, history=history):
            if isinstance(event, GenerationResult):
                await _log_conversation(session, model, question, event.answer, event.status, event.input_tokens, event.output_tokens, session_id)
                generation_cost = estimate_cost(model.generation_model, event.input_tokens, event.output_tokens)
                data = json.dumps({
                    "answer": event.answer,
                    "status": event.status,
                    "session_id": session_id,
                    "tokens_in": event.input_tokens,
                    "tokens_out": event.output_tokens,
                    "cost": f"${generation_cost + rerank_cost:.6f}",
                })
                yield f"event: done\ndata: {data}\n\n"
            else:
                yield f"data: {json.dumps(event)}\n\n"
    except anthropic.APIStatusError as e:
        if e.status_code == 529:
            error = json.dumps({"error": "AI provider is temporarily overloaded. Please try again."})
        else:
            error = json.dumps({"error": f"AI provider error ({e.status_code}). Please try again."})
        yield f"event: error\ndata: {error}\n\n"
    except Exception:
        logger.exception("Unhandled error in stream for model_id=%s", model.id)
        error = json.dumps({"error": "An unexpected error occurred. Please try again."})
        yield f"event: error\ndata: {error}\n\n"
