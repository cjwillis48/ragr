import json
import logging
import time
import uuid

import anthropic
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session, get_session
from app.dependencies import require_chat_auth
from app.services.rate_limit import RateLimiter
from app.models.conversation import Conversation, Message
from app.models.rag_model import RagModel
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.budget import check_budget, estimate_cost, estimate_rerank_cost, record_usage
from app.services.generation import GenerationResult, generate_answer, generate_answer_stream
from app.services.retrieval import ChunkScore, RetrievalResult, retrieve_with_threshold

_chat_limiter = RateLimiter(max_requests=settings.rate_limit_per_min, window_seconds=60)


router = APIRouter(tags=["chat"])
logger = logging.getLogger("ragr.chat")


def _resolve_client_ip(request: Request) -> str:
    """Extract the real client IP, trusting proxy headers only from known proxies."""
    direct_ip = request.client.host if request.client else "unknown"
    if not (settings.trusted_proxy_ips and direct_ip in settings.trusted_proxy_ips):
        return direct_ip

    forwarded_ip = (
        # Cloudflare Tunnel: single authoritative client IP
        request.headers.get("cf-connecting-ip")
        # Standard reverse proxies (nginx, ALB): leftmost IP is the client
        or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    )
    if not forwarded_ip:
        logger.warning("proxy_missing_forwarding_headers", extra={"proxy_ip": direct_ip})
    return forwarded_ip or direct_ip


async def _load_session_history(
    session: AsyncSession,
    model: RagModel,
    session_id: str,
) -> list[dict]:
    """Load the last N conversation turns for a session from the DB."""
    result = await session.execute(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Conversation.model_id == model.id,
            Conversation.session_id == session_id,
            Conversation.deleted_at.is_(None),
            Message.deleted_at.is_(None),
        )
        .order_by(Message.created_at.desc())
        .limit(model.history_turns)
    )
    rows = result.scalars().all()

    # Reverse to chronological order and flatten to message pairs
    history = []
    for row in reversed(rows):
        history.append({"role": "user", "content": row.question})
        history.append({"role": "assistant", "content": row.answer})
    return history


async def _log_message(
    session: AsyncSession,
    model: RagModel,
    question: str,
    answer: str,
    status: str,
    tokens_in: int,
    tokens_out: int,
    session_id: str | None = None,
    scores: list[ChunkScore] | None = None,
) -> None:
    """Record token usage and log the message under its conversation."""
    await record_usage(session, model, tokens_in, tokens_out)
    retrieved_chunks = (
        [{"chunk_id": s.chunk_id, "distance": s.distance, "rerank_score": s.rerank_score, "keyword_rank": s.keyword_rank, "retrieval_method": s.retrieval_method} for s in scores]
        if scores else None
    )

    # Find or create conversation
    effective_session_id = session_id or str(uuid.uuid4())
    result = await session.execute(
        select(Conversation).where(
            Conversation.model_id == model.id,
            Conversation.session_id == effective_session_id,
            Conversation.deleted_at.is_(None),
        )
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        conversation = Conversation(
            model_id=model.id,
            session_id=effective_session_id,
            title=question[:80],
            message_count=0,
        )
        session.add(conversation)
        await session.flush()

    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation.id)
        .values(message_count=Conversation.message_count + 1)
    )

    session.add(Message(
        conversation_id=conversation.id,
        model_id=model.id,
        question=question,
        answer=answer,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        retrieved_chunks=retrieved_chunks,
    ))
    await session.commit()


@router.post("/models/{slug}/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    model: RagModel = Depends(require_chat_auth),
    session: AsyncSession = Depends(get_session),
):
    """Query a model — public endpoint. Set stream: true for SSE."""
    client_ip = _resolve_client_ip(request)
    rate_key = f"{model.id}:{client_ip}"
    if not _chat_limiter.is_allowed(rate_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")

    if not await check_budget(session, model):
        raise HTTPException(status_code=429, detail="Model has exceeded its monthly budget")

    # Resolve session ID
    session_id = body.session_id or str(uuid.uuid4())

    t_req = time.perf_counter()
    try:
        retrieval = await retrieve_with_threshold(session, model, body.question)
    except httpx.TimeoutException:
        logger.error("embedding_timeout")
        raise HTTPException(status_code=503, detail="Embedding service timed out. Please try again.")
    except Exception:
        logger.exception("embedding_failed")
        raise HTTPException(status_code=503, detail="Embedding service unavailable. Please try again.")

    rerank_cost = estimate_rerank_cost(model.rerank_model, retrieval.rerank_tokens) if retrieval.rerank_tokens else 0.0
    logger.info("pre_stream_ready", extra={"duration_ms": round((time.perf_counter() - t_req) * 1000), "chunks": len(retrieval.chunks), "rerank_cost": rerank_cost})

    # Build history: prefer server-side session history, fall back to client-provided
    if body.session_id or not body.history:
        history = await _load_session_history(session, model, session_id)
    else:
        history = [{"role": m.role, "content": m.content} for m in body.history]

    history = history or None

    if body.stream:
        return StreamingResponse(
            _stream_response(model, body.question, retrieval.chunks, history, session_id, rerank_cost, retrieval.scores),
            media_type="text/event-stream",
        )

    try:
        result = await generate_answer(model, body.question, retrieval.chunks, history=history)
    except anthropic.APIStatusError as e:
        if e.status_code == 529:
            raise HTTPException(status_code=503, detail="AI provider is temporarily overloaded. Please try again.")
        raise
    await _log_message(session, model, body.question, result.answer, result.status, result.input_tokens, result.output_tokens, session_id, retrieval.scores)

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
    model: RagModel,
    question: str,
    chunks: list,
    history: list[dict] | None = None,
    session_id: str | None = None,
    rerank_cost: float = 0.0,
    scores: list[ChunkScore] | None = None,
):
    """SSE generator. Streams text deltas, then a final done event with metadata.

    Opens its own DB session for logging since the dependency-injected
    session may close before the stream finishes.
    """
    try:
        async with async_session() as stream_session:
            async for event in generate_answer_stream(model, question, chunks, history=history):
                if isinstance(event, GenerationResult):
                    await _log_message(stream_session, model, question, event.answer, event.status, event.input_tokens, event.output_tokens, session_id, scores)
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
        logger.exception("stream_error")
        error = json.dumps({"error": "An unexpected error occurred. Please try again."})
        yield f"event: error\ndata: {error}\n\n"
