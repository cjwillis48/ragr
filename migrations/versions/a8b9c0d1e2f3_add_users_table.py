"""Add users table for tracking Clerk users and global-key allowlist.

Backfills one row per existing distinct rag_models.owner_id with
allow_global_keys=true so pre-launch model owners aren't locked out.

Revision ID: a8b9c0d1e2f3
Revises: 474a0de3a18d
Create Date: 2026-05-03
"""

from alembic import op
import sqlalchemy as sa

revision = "a8b9c0d1e2f3"
down_revision = "474a0de3a18d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("clerk_user_id", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("first_name", sa.String(255), nullable=True),
        sa.Column("last_name", sa.String(255), nullable=True),
        sa.Column("allow_global_keys", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_clerk_user_id", "users", ["clerk_user_id"], unique=True)

    # Backfill: grandfather any pre-existing model owner so their models keep working.
    op.execute(
        sa.text(
            "INSERT INTO users (clerk_user_id, allow_global_keys) "
            "SELECT DISTINCT owner_id, true FROM rag_models "
            "WHERE deleted_at IS NULL AND owner_id <> 'UNASSIGNED' "
            "ON CONFLICT (clerk_user_id) DO NOTHING"
        )
    )


def downgrade() -> None:
    op.drop_index("ix_users_clerk_user_id", table_name="users")
    op.drop_table("users")
