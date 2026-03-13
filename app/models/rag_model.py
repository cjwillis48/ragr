from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class RagModel(Base):
    __tablename__ = "rag_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    greeting: Mapped[str] = mapped_column(Text, default="")
    placeholder: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")

    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    top_k: Mapped[int] = mapped_column(Integer, default=15)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    generation_model: Mapped[str] = mapped_column(String(100), nullable=False)
    reranker_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    rerank_model: Mapped[str] = mapped_column(String(100), default="rerank-2.5-lite")
    history_turns: Mapped[int] = mapped_column(Integer, default=10)

    hosted_chat: Mapped[bool] = mapped_column(Boolean, default=True)
    allowed_origins: Mapped[list] = mapped_column(JSONB, default=list)
    budget_limit: Mapped[float] = mapped_column(Float, default=10.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    chunks = relationship("ContentChunk", back_populates="rag_model", cascade="all, delete-orphan")
    conversations = relationship("ConversationLog", back_populates="rag_model", cascade="all, delete-orphan")
    token_usages = relationship("TokenUsage", back_populates="rag_model", cascade="all, delete-orphan")
    ingestion_sources = relationship("IngestionSource", back_populates="rag_model", cascade="all, delete-orphan")
    api_keys = relationship("ModelApiKey", back_populates="rag_model", cascade="all, delete-orphan")
