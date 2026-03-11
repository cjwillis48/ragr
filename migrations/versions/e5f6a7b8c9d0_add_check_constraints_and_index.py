"""add allowed_origins, CHECK constraints on rag_models, and composite index on content_chunks

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('rag_models', sa.Column('allowed_origins', JSONB, server_default='[]', nullable=False))

    op.create_check_constraint('ck_chunk_size_positive', 'rag_models', 'chunk_size > 0')
    op.create_check_constraint('ck_chunk_overlap_non_negative', 'rag_models', 'chunk_overlap >= 0')
    op.create_check_constraint('ck_similarity_threshold_range', 'rag_models', 'similarity_threshold >= 0 AND similarity_threshold <= 1')
    op.create_check_constraint('ck_top_k_positive', 'rag_models', 'top_k > 0')
    op.create_check_constraint('ck_budget_limit_non_negative', 'rag_models', 'budget_limit >= 0')

    op.create_index('ix_content_chunks_model_source', 'content_chunks', ['model_id', 'source_identifier'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_content_chunks_model_source', 'content_chunks')

    op.drop_constraint('ck_budget_limit_non_negative', 'rag_models')
    op.drop_constraint('ck_top_k_positive', 'rag_models')
    op.drop_constraint('ck_similarity_threshold_range', 'rag_models')
    op.drop_constraint('ck_chunk_overlap_non_negative', 'rag_models')
    op.drop_constraint('ck_chunk_size_positive', 'rag_models')
    op.drop_column('rag_models', 'allowed_origins')
