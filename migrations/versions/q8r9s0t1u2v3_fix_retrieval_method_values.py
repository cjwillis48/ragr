"""Fix retrieval_method values — reranker is not a retrieval source.

The previous migration incorrectly labeled all chunks with a rerank_score
as "rerank". The reranker just re-orders results; the actual retrieval
sources are vector (cosine), keyword (tsquery), or hybrid (both).

Revision ID: q8r9s0t1u2v3
Revises: p7q8r9s0t1u2
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa

revision = "q8r9s0t1u2v3"
down_revision = "p7q8r9s0t1u2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("""
            UPDATE messages
            SET retrieved_chunks = (
                SELECT jsonb_agg(
                    elem || jsonb_build_object(
                        'retrieval_method',
                        CASE
                            WHEN (elem->>'keyword_rank') IS NOT NULL AND (elem->>'distance')::float < 1.0 THEN 'hybrid'
                            WHEN (elem->>'keyword_rank') IS NOT NULL THEN 'keyword'
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
    # Revert to the old (buggy) values
    conn = op.get_bind()
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
