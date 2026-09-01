"""Give every reranked model a precision floor, and neighbours where positions exist.

Retrieval no longer applies the vector-distance cutoff when a reranker is
enabled — the reranker judges the full candidate set and rerank_threshold
becomes the only relevance filter. Models still at 0 would have none, so
backfill 0.35 (the floor validated on chatlie). Turn on neighbor_radius=1
where a re-ingested corpus actually has positions; elsewhere it stays 0
until the next re-ingest populates them. Restore similarity_threshold to
its default where it was zeroed during manual tuning — it is inert while
reranking but remains the fallback filter if the reranker is disabled.

Revision ID: f3g4h5i6j7k8
Revises: e2f3g4h5i6j7
Create Date: 2026-09-01
"""

from alembic import op

revision = "f3g4h5i6j7k8"
down_revision = "e2f3g4h5i6j7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE rag_models SET rerank_threshold = 0.35 WHERE rerank_threshold = 0")
    op.execute("""
        UPDATE rag_models SET neighbor_radius = 1
        WHERE neighbor_radius = 0 AND EXISTS (
            SELECT 1 FROM content_chunks c
            WHERE c.model_id = rag_models.id AND c.position > 0
        )
    """)
    op.execute("UPDATE rag_models SET similarity_threshold = 0.3 WHERE similarity_threshold = 0")


def downgrade() -> None:
    # Best-effort: only rows that still carry the backfilled values revert.
    op.execute("UPDATE rag_models SET rerank_threshold = 0 WHERE rerank_threshold = 0.35")
    op.execute("UPDATE rag_models SET neighbor_radius = 0 WHERE neighbor_radius = 1")
