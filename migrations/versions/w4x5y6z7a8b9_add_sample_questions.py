"""Add sample_questions to rag_models.

Revision ID: w4x5y6z7a8b9
Revises: v3w4x5y6z7a8
Create Date: 2026-03-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "w4x5y6z7a8b9"
down_revision = "v3w4x5y6z7a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_models",
        sa.Column("sample_questions", JSONB, server_default="[]", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("rag_models", "sample_questions")
