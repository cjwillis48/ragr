"""Add retrieved_chunks JSONB column to conversation_logs

Revision ID: j0k1l2m3n4o5
Revises: i9j0k1l2m3n4
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "j0k1l2m3n4o5"
down_revision = "i9j0k1l2m3n4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_logs",
        sa.Column("retrieved_chunks", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_logs", "retrieved_chunks")
