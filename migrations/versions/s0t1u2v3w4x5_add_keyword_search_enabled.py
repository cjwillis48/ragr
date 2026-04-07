"""Add keyword_search_enabled to rag_models.

Revision ID: s0t1u2v3w4x5
Revises: r9s0t1u2v3w4
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa

revision = "s0t1u2v3w4x5"
down_revision = "r9s0t1u2v3w4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_models",
        sa.Column("keyword_search_enabled", sa.Boolean, server_default="true", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("rag_models", "keyword_search_enabled")
