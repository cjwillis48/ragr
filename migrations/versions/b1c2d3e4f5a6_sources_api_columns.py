"""add source_url, content_type, status to ingestion_sources

Revision ID: b1c2d3e4f5a6
Revises: 45e52081bb3a
Create Date: 2026-03-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('ingestion_sources', sa.Column('source_url', sa.String(), server_default='', nullable=False))
    op.add_column('ingestion_sources', sa.Column('content_type', sa.String(32), server_default='text', nullable=False))
    op.add_column('ingestion_sources', sa.Column('status', sa.String(16), server_default='complete', nullable=False))
    op.add_column('ingestion_sources', sa.Column('embedding_cost', sa.Float(), server_default='0', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('ingestion_sources', 'embedding_cost')
    op.drop_column('ingestion_sources', 'status')
    op.drop_column('ingestion_sources', 'content_type')
    op.drop_column('ingestion_sources', 'source_url')
