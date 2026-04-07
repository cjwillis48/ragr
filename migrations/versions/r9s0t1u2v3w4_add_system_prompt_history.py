"""Add system_prompt_history table.

Revision ID: r9s0t1u2v3w4
Revises: q8r9s0t1u2v3
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa

revision = "r9s0t1u2v3w4"
down_revision = "q8r9s0t1u2v3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_prompt_history",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "model_id",
            sa.Integer,
            sa.ForeignKey("rag_models.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("prompt_text", sa.Text, nullable=False),
        sa.Column("source", sa.String(20), nullable=False),  # "manual" | "generated"
        sa.Column("input_text", sa.Text, nullable=True),  # what user typed before magic
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Backfill: snapshot every model's current system_prompt as initial history
    conn = op.get_bind()
    conn.execute(
        sa.text("""
            INSERT INTO system_prompt_history (model_id, prompt_text, source, created_at)
            SELECT id, system_prompt, 'manual', created_at
            FROM rag_models
            WHERE system_prompt IS NOT NULL AND system_prompt != ''
        """)
    )


def downgrade() -> None:
    op.drop_table("system_prompt_history")
