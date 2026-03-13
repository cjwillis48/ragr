"""Rename public_access to hosted_chat

Revision ID: i9j0k1l2m3n4
Revises: h8i9j0k1l2m3
Create Date: 2026-03-12
"""
from alembic import op

revision = "i9j0k1l2m3n4"
down_revision = "h8i9j0k1l2m3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("rag_models", "public_access", new_column_name="hosted_chat")


def downgrade() -> None:
    op.alter_column("rag_models", "hosted_chat", new_column_name="public_access")
