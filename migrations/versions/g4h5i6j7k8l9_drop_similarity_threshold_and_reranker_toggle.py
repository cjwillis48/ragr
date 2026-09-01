"""Drop similarity_threshold and reranker_enabled from rag_models.

Reranking is always on: cosine distance proved a crude relevance proxy, so the
reranker judges every candidate set and rerank_threshold is the only precision
floor. A rerank failure degrades to RRF order in code rather than via config.

Revision ID: g4h5i6j7k8l9
Revises: f3g4h5i6j7k8
Create Date: 2026-09-01
"""

from alembic import op
import sqlalchemy as sa

revision = "g4h5i6j7k8l9"
down_revision = "f3g4h5i6j7k8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("rag_models", "similarity_threshold")
    op.drop_column("rag_models", "reranker_enabled")


def downgrade() -> None:
    op.add_column(
        "rag_models",
        sa.Column("similarity_threshold", sa.Float, server_default="0.3", nullable=False),
    )
    op.add_column(
        "rag_models",
        sa.Column("reranker_enabled", sa.Boolean, server_default="true", nullable=False),
    )
