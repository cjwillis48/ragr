"""Add owner_id to rag_models

Revision ID: g7h8i9j0k1l2
Revises: f6a7b8c9d0e1
Create Date: 2026-03-11
"""
from alembic import op
import sqlalchemy as sa

revision = "g7h8i9j0k1l2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_models",
        sa.Column("owner_id", sa.String(255), nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_column("rag_models", "owner_id")
