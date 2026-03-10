"""add greeting and placeholder fields to rag_models

Revision ID: 45e52081bb3a
Revises: a1b2c3d4e5f6
Create Date: 2026-03-02 22:27:45.879068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '45e52081bb3a'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('rag_models', sa.Column('greeting', sa.Text(), server_default='', nullable=False))
    op.add_column('rag_models', sa.Column('placeholder', sa.Text(), server_default='', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('rag_models', 'placeholder')
    op.drop_column('rag_models', 'greeting')
