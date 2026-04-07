"""Make owner_id non-nullable on rag_models.

Any existing rows with NULL owner_id are assigned to the superuser (or a
placeholder) so the NOT NULL constraint can be applied safely.

Revision ID: y6z7a8b9c0d1
Revises: x5y6z7a8b9c0
Create Date: 2026-03-26
"""

from alembic import op
import sqlalchemy as sa

revision = "y6z7a8b9c0d1"
down_revision = "x5y6z7a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Backfill any NULL owner_id rows with a placeholder so the constraint succeeds.
    # In practice there should be none, but this makes the migration safe.
    op.execute(
        sa.text(
            "UPDATE rag_models SET owner_id = 'UNASSIGNED' WHERE owner_id IS NULL"
        )
    )
    op.alter_column(
        "rag_models",
        "owner_id",
        existing_type=sa.String(255),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "rag_models",
        "owner_id",
        existing_type=sa.String(255),
        nullable=True,
    )
