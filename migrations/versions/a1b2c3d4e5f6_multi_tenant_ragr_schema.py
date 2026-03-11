"""multi-tenant ragr schema

Revision ID: a1b2c3d4e5f6
Revises: 5cb2a1d46d26
Create Date: 2026-03-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy.vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '5cb2a1d46d26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create rag_models table
    op.create_table(
        'rag_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slug', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False, server_default=''),
        sa.Column('system_prompt', sa.Text(), nullable=False, server_default=''),
        sa.Column('chunk_size', sa.Integer(), nullable=False, server_default='500'),
        sa.Column('chunk_overlap', sa.Integer(), nullable=False, server_default='50'),
        sa.Column('similarity_threshold', sa.Float(), nullable=False, server_default='0.75'),
        sa.Column('top_k', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('embedding_model', sa.String(length=100), nullable=False),
        sa.Column('generation_model', sa.String(length=100), nullable=False),
        sa.Column('budget_limit', sa.Float(), nullable=False, server_default='10.0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_rag_models_slug', 'rag_models', ['slug'], unique=True)

    # Create ingestion_sources table
    op.create_table(
        'ingestion_sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('source_identifier', sa.String(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['model_id'], ['rag_models.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id', 'source_identifier', name='uq_model_source'),
    )
    op.create_index('ix_ingestion_sources_model_id', 'ingestion_sources', ['model_id'])

    # Add model_id to content_chunks
    op.add_column('content_chunks', sa.Column('model_id', sa.Integer(), nullable=True))
    op.add_column('content_chunks', sa.Column('source_identifier', sa.String(), nullable=False, server_default=''))
    op.create_index('ix_content_chunks_model_id', 'content_chunks', ['model_id'])
    op.create_foreign_key(
        'fk_content_chunks_model_id', 'content_chunks', 'rag_models',
        ['model_id'], ['id'], ondelete='CASCADE'
    )

    # Add model_id to conversation_logs
    op.add_column('conversation_logs', sa.Column('model_id', sa.Integer(), nullable=True))
    op.create_index('ix_conversation_logs_model_id', 'conversation_logs', ['model_id'])
    op.create_foreign_key(
        'fk_conversation_logs_model_id', 'conversation_logs', 'rag_models',
        ['model_id'], ['id'], ondelete='CASCADE'
    )

    # Add model_id to token_usage, change unique constraint
    op.add_column('token_usage', sa.Column('model_id', sa.Integer(), nullable=True))
    op.drop_constraint('token_usage_month_key', 'token_usage', type_='unique')
    op.create_unique_constraint('uq_model_month', 'token_usage', ['model_id', 'month'])
    op.create_index('ix_token_usage_model_id', 'token_usage', ['model_id'])
    op.create_foreign_key(
        'fk_token_usage_model_id', 'token_usage', 'rag_models',
        ['model_id'], ['id'], ondelete='CASCADE'
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove model_id from token_usage
    op.drop_constraint('fk_token_usage_model_id', 'token_usage', type_='foreignkey')
    op.drop_index('ix_token_usage_model_id', table_name='token_usage')
    op.drop_constraint('uq_model_month', 'token_usage', type_='unique')
    op.create_unique_constraint('token_usage_month_key', 'token_usage', ['month'])
    op.drop_column('token_usage', 'model_id')

    # Remove model_id from conversation_logs
    op.drop_constraint('fk_conversation_logs_model_id', 'conversation_logs', type_='foreignkey')
    op.drop_index('ix_conversation_logs_model_id', table_name='conversation_logs')
    op.drop_column('conversation_logs', 'model_id')

    # Remove model_id + source_identifier from content_chunks
    op.drop_constraint('fk_content_chunks_model_id', 'content_chunks', type_='foreignkey')
    op.drop_index('ix_content_chunks_model_id', table_name='content_chunks')
    op.drop_column('content_chunks', 'source_identifier')
    op.drop_column('content_chunks', 'model_id')

    # Drop new tables
    op.drop_index('ix_ingestion_sources_model_id', table_name='ingestion_sources')
    op.drop_table('ingestion_sources')
    op.drop_index('ix_rag_models_slug', table_name='rag_models')
    op.drop_table('rag_models')
