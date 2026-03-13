"""Add session_id to conversation_logs and history_turns to rag_models

Revision ID: h8i9j0k1l2m3
Revises: g7h8i9j0k1l2
Create Date: 2026-03-12
"""
from alembic import op
import sqlalchemy as sa

revision = "h8i9j0k1l2m3"
down_revision = "g7h8i9j0k1l2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_logs",
        sa.Column("session_id", sa.String(64), nullable=True, index=True),
    )
    op.add_column(
        "rag_models",
        sa.Column("history_turns", sa.Integer(), server_default="10", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("rag_models", "history_turns")
    op.drop_column("conversation_logs", "session_id")
