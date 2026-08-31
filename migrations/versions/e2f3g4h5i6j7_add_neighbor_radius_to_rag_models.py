"""Add neighbor_radius to rag_models.

Retrieval returns individual chunks, so an answer that straddles a chunk
boundary arrives half-missing. With a position column on content_chunks we can
pull each hit's neighbours back in. That multiplies the context sent to the
model, so it is opt-in per model and off by default.

Revision ID: e2f3g4h5i6j7
Revises: d1e2f3g4h5i6
Create Date: 2026-08-31
"""

from alembic import op
import sqlalchemy as sa

revision = "e2f3g4h5i6j7"
down_revision = "d1e2f3g4h5i6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_models",
        sa.Column("neighbor_radius", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("rag_models", "neighbor_radius")
