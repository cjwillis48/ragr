"""add model_api_keys table and public_access column

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-03-09
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, Sequence[str], None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'model_api_keys',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('rag_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('label', sa.String(255), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('key_prefix', sa.String(12), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_model_api_keys_model_id', 'model_api_keys', ['model_id'])

    op.add_column('rag_models', sa.Column('public_access', sa.Boolean(), server_default='true', nullable=False))


def downgrade() -> None:
    op.drop_column('rag_models', 'public_access')
    op.drop_index('ix_model_api_keys_model_id', 'model_api_keys')
    op.drop_table('model_api_keys')
