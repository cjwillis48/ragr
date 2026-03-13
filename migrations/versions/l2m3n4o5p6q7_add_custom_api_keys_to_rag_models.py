"""Add custom_anthropic_key and custom_voyage_key to rag_models

Revision ID: l2m3n4o5p6q7
Revises: k1l2m3n4o5p6
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa

revision = "l2m3n4o5p6q7"
down_revision = "k1l2m3n4o5p6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("rag_models", sa.Column("custom_anthropic_key", sa.Text(), nullable=True))
    op.add_column("rag_models", sa.Column("custom_voyage_key", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("rag_models", "custom_voyage_key")
    op.drop_column("rag_models", "custom_anthropic_key")
