"""Add rerank_candidates to rag_models.

Revision ID: u2v3w4x5y6z7
Revises: t1u2v3w4x5y6
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa

revision = "u2v3w4x5y6z7"
down_revision = "t1u2v3w4x5y6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_models",
        sa.Column("rerank_candidates", sa.Integer, server_default="60", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("rag_models", "rerank_candidates")
