"""Add position to content_chunks.

Chunks had no ordering column, so the only way to recover a source's chunk
order was the surrogate id. Recording the ordinal lets retrieval fetch a hit's
neighbours and re-join context that chunking split apart, and gives the
`metadata` JSONB column (previously always `{}`) something to hang offsets and
heading paths off.

Existing rows default to 0: they predate structure-aware chunking and are
re-chunked rather than backfilled.

Revision ID: d1e2f3g4h5i6
Revises: c0d1e2f3g4h5
Create Date: 2026-08-31
"""

from alembic import op
import sqlalchemy as sa

revision = "d1e2f3g4h5i6"
down_revision = "c0d1e2f3g4h5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "content_chunks",
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_content_chunks_model_source_position",
        "content_chunks",
        ["model_id", "source_identifier", "position"],
    )
    # (model_id, source_identifier) is a strict prefix of the index above, so
    # Postgres can serve those lookups from it. Keeping both just costs writes.
    op.drop_index("ix_content_chunks_model_source", table_name="content_chunks")


def downgrade() -> None:
    op.create_index(
        "ix_content_chunks_model_source", "content_chunks", ["model_id", "source_identifier"]
    )
    op.drop_index("ix_content_chunks_model_source_position", table_name="content_chunks")
    op.drop_column("content_chunks", "position")
