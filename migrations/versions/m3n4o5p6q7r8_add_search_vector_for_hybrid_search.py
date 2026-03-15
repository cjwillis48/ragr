"""Add search_vector tsvector column and GIN index for hybrid search

Revision ID: m3n4o5p6q7r8
Revises: l2m3n4o5p6q7
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR

revision = "m3n4o5p6q7r8"
down_revision = "l2m3n4o5p6q7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("content_chunks", sa.Column("search_vector", TSVECTOR, nullable=True))
    op.create_index(
        "ix_content_chunks_search_vector",
        "content_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    # Backfill existing chunks
    op.execute("""
        UPDATE content_chunks
        SET search_vector = to_tsvector('english', content)
        WHERE search_vector IS NULL
    """)


def downgrade() -> None:
    op.drop_index("ix_content_chunks_search_vector", table_name="content_chunks")
    op.drop_column("content_chunks", "search_vector")
