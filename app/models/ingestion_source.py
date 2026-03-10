from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class IngestionSource(Base):
    __tablename__ = "ingestion_sources"
    __table_args__ = (
        UniqueConstraint("model_id", "source_identifier", name="uq_model_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rag_models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_identifier: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    source_url: Mapped[str] = mapped_column(String, server_default="", nullable=False)
    content_type: Mapped[str] = mapped_column(String(32), server_default="text", nullable=False)
    status: Mapped[str] = mapped_column(String(16), server_default="complete", nullable=False)
    embedding_cost: Mapped[float] = mapped_column(server_default="0", nullable=False)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    rag_model = relationship("RagModel", back_populates="ingestion_sources")
