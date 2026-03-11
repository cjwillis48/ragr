"""add reranker_enabled and rerank_model to rag_models

Revision ID: d4e5f6a7b8c9
Revises: b1c2d3e4f5a6
Create Date: 2026-03-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('rag_models', sa.Column('reranker_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('rag_models', sa.Column('rerank_model', sa.String(100), server_default='rerank-2.5-lite', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('rag_models', 'rerank_model')
    op.drop_column('rag_models', 'reranker_enabled')
