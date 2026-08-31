"""Add status_detail to ingestion_sources.

A crawl that finds nothing marks the root source "failed" with no explanation.
The crawler knows why (non-HTML, oversized, or — most often — the page rendered
no text because the site is a client-side JavaScript app), but that diagnosis
was discarded. This column gives it somewhere to live so the console can tell
the user what actually happened and what to do instead.

Revision ID: c0d1e2f3g4h5
Revises: b9c0d1e2f3g4
Create Date: 2026-08-30
"""

from alembic import op
import sqlalchemy as sa

revision = "c0d1e2f3g4h5"
down_revision = "b9c0d1e2f3g4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingestion_sources",
        sa.Column("status_detail", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingestion_sources", "status_detail")
