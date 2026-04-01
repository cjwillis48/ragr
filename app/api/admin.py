import json
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import Date, cast, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_session
from app.dependencies import require_model_auth
from app.models.content import ContentChunk
from app.models.conversation import Conversation, Message
from app.models.ingestion_source import IngestionSource
from app.models.rag_model import RagModel
from app.models.system_prompt_history import SystemPromptHistory
from app.schemas.admin import ChunkResponse, ConversationDetailResponse, ConversationListResponse, \
    ConversationSummaryResponse, DailyStatsEntry, StatsResponse, SystemPromptHistoryResponse, TopSourceEntry
from app.services.budget import get_current_month_usage

router = APIRouter(tags=["admin"])
logger = logging.getLogger("ragr.admin")


@router.get(
    "/models/{slug}/stats",
    response_model=StatsResponse,
)
async def model_stats(
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
):
    """Get per-model statistics."""
    chunk_count = await session.scalar(
        select(func.count()).select_from(ContentChunk).where(ContentChunk.model_id == model.id)
    )

    convo_count = await session.scalar(
        select(func.count()).select_from(Conversation).where(
            Conversation.model_id == model.id, Conversation.deleted_at.is_(None))
    )

    message_count = await session.scalar(
        select(func.count()).select_from(Message).where(
            Message.model_id == model.id, Message.deleted_at.is_(None))
    )

    unanswered = await session.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.model_id == model.id, Message.status == "unanswered", Message.deleted_at.is_(None))
    )

    source_count = await session.scalar(
        select(func.count()).select_from(IngestionSource).where(IngestionSource.model_id == model.id)
    )

    usage = await get_current_month_usage(session, model)
    current_cost = usage.estimated_cost if usage else 0.0

    return StatsResponse(
        model_slug=model.slug,
        total_chunks=chunk_count or 0,
        total_conversations=convo_count or 0,
        total_messages=message_count or 0,
        unanswered_questions=unanswered or 0,
        current_month_cost=round(current_cost, 4),
        budget_limit=model.budget_limit,
        budget_remaining=round(model.budget_limit - current_cost, 4),
        total_sources=source_count or 0,
    )


@router.get(
    "/models/{slug}/stats/daily",
    response_model=list[DailyStatsEntry],
)
async def daily_stats(
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
        days: int = Query(30, ge=1, le=365),
):
    """Daily message stats for the last N days."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    day_col = cast(Message.created_at, Date).label("day")

    result = await session.execute(
        select(
            day_col,
            func.count().filter(Message.status == "answered").label("answered"),
            func.count().filter(Message.status == "unanswered").label("unanswered"),
            func.count().filter(Message.status == "off_topic").label("off_topic"),
            func.coalesce(func.sum(Message.tokens_in), 0).label("tokens_in"),
            func.coalesce(func.sum(Message.tokens_out), 0).label("tokens_out"),
        )
        .where(
            Message.model_id == model.id,
            Message.deleted_at.is_(None),
            Message.created_at >= cutoff,
        )
        .group_by(day_col)
        .order_by(day_col)
    )
    rows = result.all()

    return [
        DailyStatsEntry(
            date=row.day,
            answered=row.answered,
            unanswered=row.unanswered,
            off_topic=row.off_topic,
            tokens_in=row.tokens_in,
            tokens_out=row.tokens_out,
        )
        for row in rows
    ]


@router.get(
    "/models/{slug}/stats/top-sources",
    response_model=list[TopSourceEntry],
)
async def top_sources(
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
        limit: int = Query(10, ge=1, le=50),
):
    """Top sources by retrieval count. Counts how often each source's chunks appear in retrieved_chunks."""
    result = await session.execute(
        text("""
            SELECT cc.source_identifier,
                   COUNT(*) AS retrieval_count,
                   COALESCE(src.chunk_count, 0) AS chunk_count
            FROM messages m
            CROSS JOIN LATERAL jsonb_array_elements(m.retrieved_chunks) AS elem
            JOIN content_chunks cc ON cc.id = (elem->>'chunk_id')::int
            LEFT JOIN ingestion_sources src
                ON src.model_id = m.model_id AND src.source_identifier = cc.source_identifier
            WHERE m.model_id = :model_id
              AND m.deleted_at IS NULL
              AND m.retrieved_chunks IS NOT NULL
              AND jsonb_typeof(m.retrieved_chunks) = 'array'
            GROUP BY cc.source_identifier, src.chunk_count
            ORDER BY retrieval_count DESC
            LIMIT :limit
        """),
        {"model_id": model.id, "limit": limit},
    )

    return [
        TopSourceEntry(
            source_identifier=row.source_identifier,
            retrieval_count=row.retrieval_count,
            chunk_count=row.chunk_count,
        )
        for row in result
    ]


@router.get(
    "/models/{slug}/conversations",
    response_model=ConversationListResponse,
)
async def list_conversations(
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
):
    """List conversations for a model, newest first."""
    base = select(Conversation).where(Conversation.model_id == model.id, Conversation.deleted_at.is_(None))

    total = await session.scalar(
        select(func.count()).select_from(base.subquery())
    )

    result = await session.execute(
        base.order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    convos = result.scalars().all()

    return ConversationListResponse(
        model_slug=model.slug,
        conversations=[ConversationSummaryResponse.model_validate(c) for c in convos],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/models/{slug}/conversations/{conversation_id}/messages",
    response_model=ConversationDetailResponse,
)
async def get_conversation_messages(
        conversation_id: int,
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
):
    """Get all messages for a specific conversation."""
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.model_id == model.id, Conversation.deleted_at.is_(None))
        .options(selectinload(Conversation.messages))
    )
    convo = result.scalar_one_or_none()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Filter out soft-deleted messages
    convo_dict = {
        "id": convo.id,
        "session_id": convo.session_id,
        "title": convo.title,
        "message_count": convo.message_count,
        "created_at": convo.created_at,
        "updated_at": convo.updated_at,
        "messages": [m for m in convo.messages if m.deleted_at is None],
    }
    return ConversationDetailResponse.model_validate(convo_dict)


@router.delete(
    "/models/{slug}/conversations/{conversation_id}",
    status_code=204,
)
async def delete_conversation(
        conversation_id: int,
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
):
    """Soft-delete a conversation and its messages."""
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.model_id == model.id, Conversation.deleted_at.is_(None))
    )
    convo = result.scalar_one_or_none()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    convo.deleted_at = func.now()
    await session.execute(
        update(Message)
        .where(Message.conversation_id == convo.id, Message.deleted_at.is_(None))
        .values(deleted_at=func.now())
    )
    await session.commit()


@router.get(
    "/models/{slug}/chunks",
    response_model=list[ChunkResponse],
)
async def get_chunks(
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
        ids: str = Query(..., description="Comma-separated chunk IDs"),
):
    """Fetch chunks by ID for a model. Used to inspect retrieved context for a conversation."""
    try:
        chunk_ids = [int(i) for i in ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be comma-separated integers")

    if len(chunk_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 chunk IDs per request")

    result = await session.execute(
        select(ContentChunk).where(
            ContentChunk.model_id == model.id,
            ContentChunk.id.in_(chunk_ids),
        )
    )
    return result.scalars().all()


# --- System Prompt History & Generation ---


@router.get(
    "/models/{slug}/system-prompt-history",
    response_model=list[SystemPromptHistoryResponse],
)
async def list_system_prompt_history(
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
):
    """List system prompt history, newest first."""
    result = await session.execute(
        select(SystemPromptHistory)
        .where(SystemPromptHistory.model_id == model.id)
        .order_by(SystemPromptHistory.created_at.desc())
    )
    return [SystemPromptHistoryResponse.model_validate(r) for r in result.scalars().all()]


@router.post(
    "/models/{slug}/system-prompt-history/{history_id}/rollback",
    response_model=SystemPromptHistoryResponse,
)
async def rollback_system_prompt(
        history_id: int,
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
):
    """Rollback to a previous system prompt. Creates a new history entry."""
    result = await session.execute(
        select(SystemPromptHistory).where(
            SystemPromptHistory.id == history_id,
            SystemPromptHistory.model_id == model.id,
        )
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found")

    # Record the rollback as a new history entry
    new_entry = SystemPromptHistory(
        model_id=model.id,
        prompt_text=entry.prompt_text,
        source="manual",
        input_text=f"Rolled back to version from {entry.created_at.isoformat()}",
    )
    session.add(new_entry)

    model.system_prompt = entry.prompt_text
    await session.commit()
    await session.refresh(new_entry)
    return SystemPromptHistoryResponse.model_validate(new_entry)


_SYSTEM_PROMPT_GENERATOR = """You are an expert at writing system prompts for RAG (Retrieval-Augmented Generation) chatbots.

Given the bot's name, description, and optionally the user's rough draft or notes, write an effective system prompt.

Guidelines:
- Write in second person ("You are...")
- Be specific about the bot's domain, tone, and boundaries
- Include guidance on how to handle off-topic questions
- Keep it concise but thorough (aim for 3-8 sentences)
- Don't include RAG-specific instructions (those are added automatically)
- Match the tone to the domain (professional for business, friendly for community, etc.)

Respond with ONLY the system prompt text, no explanations or markdown."""


class GenerateSystemPromptRequest(BaseModel):
    input_text: str = ""


@router.post("/models/{slug}/generate-system-prompt")
async def generate_system_prompt(
        body: GenerateSystemPromptRequest,
        model: RagModel = Depends(require_model_auth),
):
    """Stream-generate a system prompt based on model info and user input."""
    user_parts = [f"Bot name: {model.name}"]
    if model.description:
        user_parts.append(f"Bot description: {model.description}")
    if body.input_text.strip():
        user_parts.append(f"User's notes/draft:\n{body.input_text.strip()}")

    user_message = "\n\n".join(user_parts)

    from app.services.generation import _get_client
    client = _get_client(model.custom_anthropic_key)

    async def stream():
        try:
            async with client.messages.stream(
                    model="claude-haiku-4-5",
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT_GENERATOR,
                    messages=[{"role": "user", "content": user_message}],
            ) as response:
                async for text in response.text_stream:
                    yield f"data: {json.dumps(text)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception:
            logger.exception("System prompt generation failed for model_id=%s", model.id)
            yield f"event: error\ndata: {json.dumps({'error': 'Generation failed'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class AcceptGeneratedPromptRequest(BaseModel):
    prompt_text: str
    input_text: str = ""


@router.post(
    "/models/{slug}/system-prompt-history/accept-generated",
    response_model=SystemPromptHistoryResponse,
)
async def accept_generated_prompt(
        body: AcceptGeneratedPromptRequest,
        model: RagModel = Depends(require_model_auth),
        session: AsyncSession = Depends(get_session),
):
    """Accept a generated system prompt — saves it and records history."""
    entry = SystemPromptHistory(
        model_id=model.id,
        prompt_text=body.prompt_text,
        source="generated",
        input_text=body.input_text or None,
    )
    session.add(entry)
    model.system_prompt = body.prompt_text
    await session.commit()
    await session.refresh(entry)
    return SystemPromptHistoryResponse.model_validate(entry)


# --- Sample Questions Generation ---

_SAMPLE_QUESTIONS_GENERATOR = """You are generating sample questions for a RAG chatbot.

Given the bot's name, description, system prompt, and a sample of its knowledge base content, generate 3 questions that a first-time user would likely ask. The questions should:
- Cover different topics from the knowledge base
- Be natural and conversational
- Show the range of what the bot can answer
- Be specific enough to get good answers (not generic like "tell me everything")

Respond with ONLY a JSON array of strings, no explanations:
["Question 1?", "Question 2?", "Question 3?"]"""


@router.post(
    "/models/{slug}/generate-sample-questions",
    response_model=list[str],
)
async def generate_sample_questions(
    model: RagModel = Depends(require_model_auth),
    session: AsyncSession = Depends(get_session),
):
    """Generate sample questions based on the model's knowledge base."""
    # Grab a sample of chunk content to show the LLM what the KB covers
    result = await session.execute(
        select(ContentChunk.content)
        .where(ContentChunk.model_id == model.id)
        .order_by(func.random())
        .limit(10)
    )
    sample_chunks = [row[0][:500] for row in result]

    user_parts = [f"Bot name: {model.name}"]
    if model.description:
        user_parts.append(f"Bot description: {model.description}")
    if model.system_prompt:
        user_parts.append(f"System prompt: {model.system_prompt}")
    if sample_chunks:
        user_parts.append(f"Sample knowledge base content:\n" + "\n---\n".join(sample_chunks))

    user_message = "\n\n".join(user_parts)

    from app.services.generation import _get_client
    client = _get_client(model.custom_anthropic_key)

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=_SAMPLE_QUESTIONS_GENERATOR,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        parsed = json.loads(raw)
        # Support both {"questions": [...]} and bare [...]
        questions = parsed if isinstance(parsed, list) else parsed.get("questions", [])
        return [q for q in questions if isinstance(q, str)][:3]
    except (json.JSONDecodeError, IndexError, KeyError):
        logger.warning("Failed to parse sample questions response: %s", raw)

    raise HTTPException(status_code=500, detail="Failed to generate sample questions")
