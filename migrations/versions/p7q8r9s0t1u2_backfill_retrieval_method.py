"""Backfill retrieval_method into messages.retrieved_chunks JSONB.

Revision ID: p7q8r9s0t1u2
Revises: o6p7q8r9s0t1
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa

revision = "p7q8r9s0t1u2"
down_revision = "o6p7q8r9s0t1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # Update each element in the retrieved_chunks JSONB array with a retrieval_method field.
    # NOTE: This migration had a bug — it labeled everything with rerank_score as "rerank".
    # Fixed by migration q8r9s0t1u2v3.
    conn.execute(
        sa.text("""
            UPDATE messages
            SET retrieved_chunks = (
                SELECT jsonb_agg(
                    elem || jsonb_build_object(
                        'retrieval_method',
                        CASE
                            WHEN (elem->>'rerank_score') IS NOT NULL THEN 'rerank'
                            WHEN (elem->>'keyword_rank') IS NOT NULL AND (elem->>'distance')::float >= 1.0 THEN 'keyword'
                            WHEN (elem->>'keyword_rank') IS NOT NULL THEN 'hybrid'
                            ELSE 'vector'
                        END
                    )
                )
                FROM jsonb_array_elements(retrieved_chunks) AS elem
            )
            WHERE retrieved_chunks IS NOT NULL
              AND jsonb_typeof(retrieved_chunks) = 'array'
        """)
    )


def downgrade() -> None:
    conn = op.get_bind()
    # Strip the retrieval_method key from each element
    conn.execute(
        sa.text("""
            UPDATE messages
            SET retrieved_chunks = (
                SELECT jsonb_agg(elem - 'retrieval_method')
                FROM jsonb_array_elements(retrieved_chunks) AS elem
            )
            WHERE retrieved_chunks IS NOT NULL
              AND jsonb_typeof(retrieved_chunks) = 'array'
        """)
    )
