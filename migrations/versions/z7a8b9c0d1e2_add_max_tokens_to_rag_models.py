"""Add max_tokens column to rag_models.

Revision ID: z7a8b9c0d1e2
Revises: y6z7a8b9c0d1
Create Date: 2026-04-01
"""

from alembic import op
import sqlalchemy as sa

revision = "z7a8b9c0d1e2"
down_revision = "y6z7a8b9c0d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("rag_models", sa.Column("max_tokens", sa.Integer(), nullable=False, server_default="1024"))


def downgrade() -> None:
    op.drop_column("rag_models", "max_tokens")
