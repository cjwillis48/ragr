"""Add chat_theme JSONB column, migrate greeting/placeholder, drop old columns

Revision ID: k1l2m3n4o5p6
Revises: j0k1l2m3n4o5
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "k1l2m3n4o5p6"
down_revision = "j0k1l2m3n4o5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("rag_models", sa.Column("chat_theme", JSONB, nullable=True))

    # Migrate existing greeting/placeholder values into chat_theme
    op.execute("""
        UPDATE rag_models
        SET chat_theme = jsonb_build_object(
            'greeting', CASE WHEN greeting != '' THEN greeting ELSE NULL END,
            'placeholder', CASE WHEN placeholder != '' THEN placeholder ELSE NULL END
        )
        WHERE greeting != '' OR placeholder != ''
    """)

    op.drop_column("rag_models", "greeting")
    op.drop_column("rag_models", "placeholder")


def downgrade() -> None:
    op.add_column("rag_models", sa.Column("greeting", sa.Text(), server_default="", nullable=False))
    op.add_column("rag_models", sa.Column("placeholder", sa.Text(), server_default="", nullable=False))

    # Restore values from chat_theme
    op.execute("""
        UPDATE rag_models
        SET greeting = COALESCE(chat_theme->>'greeting', ''),
            placeholder = COALESCE(chat_theme->>'placeholder', '')
        WHERE chat_theme IS NOT NULL
    """)

    op.drop_column("rag_models", "chat_theme")
