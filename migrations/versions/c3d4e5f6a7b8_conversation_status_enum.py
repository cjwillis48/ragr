"""replace answered/deflected booleans with status enum on conversation_logs

Revision ID: c3d4e5f6a7b8
Revises: 45e52081bb3a
Create Date: 2026-03-03 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = '45e52081bb3a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'conversation_logs',
        sa.Column('status', sa.String(20), nullable=False, server_default='answered'),
    )
    # Backfill existing rows from the old answered bool
    op.execute(
        "UPDATE conversation_logs SET status = CASE WHEN answered = true THEN 'answered' ELSE 'unanswered' END"
    )
    op.drop_column('conversation_logs', 'answered')
    op.drop_column('conversation_logs', 'flagged_unanswered')
    op.drop_column('conversation_logs', 'deflected')


def downgrade() -> None:
    op.add_column('conversation_logs', sa.Column('answered', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('conversation_logs', sa.Column('flagged_unanswered', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('conversation_logs', sa.Column('deflected', sa.Boolean(), nullable=False, server_default='false'))
    op.execute(
        "UPDATE conversation_logs SET answered = (status = 'answered'), "
        "flagged_unanswered = (status != 'answered'), deflected = (status != 'answered')"
    )
    op.drop_column('conversation_logs', 'status')
