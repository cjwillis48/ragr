"""Raise the rerank floor from 0.35 to a value that actually floors something.

With the distance wall gone, rerank_threshold is the only relevance filter left —
so it matters that 0.35 was never filtering anything. rerank-2.5-lite returns
scores in roughly the 0.34-0.70 band, which puts 0.35 underneath the model's
entire operating range.

Measured over two corpora of deliberately different shape (147 chunks of prose,
579 of source code), sweeping the threshold:

    0.35   recall 1.000   abstained 0.000   14.5 chunks kept for an unanswerable question
    0.45   recall 1.000   abstained 0.286    2.5
    0.50   recall 0.964   abstained 0.643    0.6

0.45 costs no recall on either corpus and cuts the junk reaching the model by
~5.8x. 0.48 scored better still, but the prose corpus loses its first query at
0.50 — tuning a platform default that close to a cliff, on a 28-query sample,
is fitting to noise.

Only models sitting on the old default are moved. Anything deliberately tuned to
another value is left alone.

Revision ID: j7k8l9m0n1o2
Revises: g4h5i6j7k8l9
Create Date: 2026-09-02
"""

from alembic import op

revision = "j7k8l9m0n1o2"
down_revision = "g4h5i6j7k8l9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE rag_models SET rerank_threshold = 0.45 WHERE rerank_threshold = 0.35")


def downgrade() -> None:
    op.execute("UPDATE rag_models SET rerank_threshold = 0.35 WHERE rerank_threshold = 0.45")
