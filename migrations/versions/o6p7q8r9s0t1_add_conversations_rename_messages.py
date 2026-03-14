"""Add conversations table and rename conversation_logs to messages.

Revision ID: o6p7q8r9s0t1
Revises: n4o5p6q7r8s9
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "o6p7q8r9s0t1"
down_revision = "n4o5p6q7r8s9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "model_id",
            sa.Integer,
            sa.ForeignKey("rag_models.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("session_id", sa.String(64), nullable=False, index=True),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("message_count", sa.Integer, server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("model_id", "session_id", name="uq_conversations_model_session"),
    )

    # 2. Add conversation_id column (nullable for now)
    op.add_column(
        "conversation_logs",
        sa.Column("conversation_id", sa.Integer, nullable=True),
    )

    # 3. Backfill: create Conversation rows from existing data
    conn = op.get_bind()

    # 3a. Rows WITH session_id: group by (model_id, session_id)
    conn.execute(
        sa.text("""
            INSERT INTO conversations (model_id, session_id, title, message_count, created_at, updated_at)
            SELECT
                cl.model_id,
                cl.session_id,
                LEFT(first_q.question, 80),
                cnt.c,
                MIN(cl.created_at),
                MAX(cl.created_at)
            FROM conversation_logs cl
            JOIN (
                SELECT model_id, session_id, COUNT(*) AS c
                FROM conversation_logs
                WHERE session_id IS NOT NULL
                GROUP BY model_id, session_id
            ) cnt ON cnt.model_id = cl.model_id AND cnt.session_id = cl.session_id
            JOIN LATERAL (
                SELECT question FROM conversation_logs sub
                WHERE sub.model_id = cl.model_id AND sub.session_id = cl.session_id
                ORDER BY sub.created_at ASC LIMIT 1
            ) first_q ON true
            WHERE cl.session_id IS NOT NULL
            GROUP BY cl.model_id, cl.session_id, first_q.question, cnt.c
        """)
    )

    # 3b. Rows with NULL session_id: each gets its own conversation
    conn.execute(
        sa.text("""
            INSERT INTO conversations (model_id, session_id, title, message_count, created_at, updated_at)
            SELECT
                model_id,
                gen_random_uuid()::text,
                LEFT(question, 80),
                1,
                created_at,
                created_at
            FROM conversation_logs
            WHERE session_id IS NULL
        """)
    )

    # 3c. Backfill conversation_id for rows WITH session_id
    conn.execute(
        sa.text("""
            UPDATE conversation_logs cl
            SET conversation_id = c.id
            FROM conversations c
            WHERE cl.session_id IS NOT NULL
              AND c.model_id = cl.model_id
              AND c.session_id = cl.session_id
        """)
    )

    # 3d. Backfill conversation_id for rows with NULL session_id (match by title + created_at)
    conn.execute(
        sa.text("""
            UPDATE conversation_logs cl
            SET conversation_id = c.id
            FROM conversations c
            WHERE cl.session_id IS NULL
              AND c.model_id = cl.model_id
              AND c.title = LEFT(cl.question, 80)
              AND c.created_at = cl.created_at
              AND c.message_count = 1
        """)
    )

    # 4. Make conversation_id NOT NULL, add FK + index
    op.alter_column("conversation_logs", "conversation_id", nullable=False)
    op.create_foreign_key(
        "fk_conversation_logs_conversation_id",
        "conversation_logs",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_conversation_logs_conversation_id",
        "conversation_logs",
        ["conversation_id"],
    )

    # 5. Rename table
    op.rename_table("conversation_logs", "messages")

    # 6. Drop session_id from messages (now on conversations)
    op.drop_index("ix_conversation_logs_session_id", table_name="messages")
    op.drop_column("messages", "session_id")


def downgrade() -> None:
    # Reverse: add session_id back, rename table, drop conversations
    op.add_column(
        "messages",
        sa.Column("session_id", sa.String(64), nullable=True),
    )

    # Backfill session_id from conversations
    conn = op.get_bind()
    conn.execute(
        sa.text("""
            UPDATE messages m
            SET session_id = c.session_id
            FROM conversations c
            WHERE m.conversation_id = c.id
        """)
    )

    op.create_index("ix_conversation_logs_session_id", "messages", ["session_id"])

    # Rename back
    op.rename_table("messages", "conversation_logs")

    # Drop FK and index
    op.drop_index("ix_conversation_logs_conversation_id", table_name="conversation_logs")
    op.drop_constraint("fk_conversation_logs_conversation_id", "conversation_logs", type_="foreignkey")
    op.drop_column("conversation_logs", "conversation_id")

    # Drop conversations table
    op.drop_table("conversations")
