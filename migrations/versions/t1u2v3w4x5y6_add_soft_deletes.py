"""Add soft deletes to rag_models, conversations, and messages.

Revision ID: t1u2v3w4x5y6
Revises: s0t1u2v3w4x5
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa

revision = "t1u2v3w4x5y6"
down_revision = "s0t1u2v3w4x5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("rag_models", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("conversations", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("messages", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index("ix_rag_models_deleted_at", "rag_models", ["deleted_at"])
    op.create_index("ix_conversations_deleted_at", "conversations", ["deleted_at"])
    op.create_index("ix_messages_deleted_at", "messages", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_messages_deleted_at", table_name="messages")
    op.drop_index("ix_conversations_deleted_at", table_name="conversations")
    op.drop_index("ix_rag_models_deleted_at", table_name="rag_models")

    op.drop_column("messages", "deleted_at")
    op.drop_column("conversations", "deleted_at")
    op.drop_column("rag_models", "deleted_at")
