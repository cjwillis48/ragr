"""Add rerank_threshold to rag_models.

Revision ID: v3w4x5y6z7a8
Revises: u2v3w4x5y6z7
Create Date: 2026-03-20
"""

from alembic import op
import sqlalchemy as sa

revision = "v3w4x5y6z7a8"
down_revision = "u2v3w4x5y6z7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_models",
        sa.Column("rerank_threshold", sa.Float, server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("rag_models", "rerank_threshold")
