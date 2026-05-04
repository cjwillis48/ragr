"""Rename question/answer to message/response (and related field renames).

- messages.question  -> messages.message
- messages.answer    -> messages.response
- rag_models.sample_questions -> rag_models.sample_messages
- chat_theme JSONB key show_sample_questions_in_greeting -> show_sample_messages_in_greeting

The "question/answer" terms misled — user inputs aren't always questions, and
"answer" implied factual correctness we can't promise. RENAME COLUMN is a fast
metadata-only operation in Postgres, so this is safe even on large tables.

Revision ID: b9c0d1e2f3g4
Revises: a8b9c0d1e2f3
Create Date: 2026-05-04
"""

from alembic import op
import sqlalchemy as sa

revision = "b9c0d1e2f3g4"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("messages", "question", new_column_name="message")
    op.alter_column("messages", "answer", new_column_name="response")
    op.alter_column("rag_models", "sample_questions", new_column_name="sample_messages")
    op.execute(sa.text(
        "UPDATE rag_models "
        "SET chat_theme = (chat_theme - 'show_sample_questions_in_greeting') "
        "                 || jsonb_build_object('show_sample_messages_in_greeting', "
        "                                       chat_theme->'show_sample_questions_in_greeting') "
        "WHERE chat_theme ? 'show_sample_questions_in_greeting'"
    ))


def downgrade() -> None:
    op.execute(sa.text(
        "UPDATE rag_models "
        "SET chat_theme = (chat_theme - 'show_sample_messages_in_greeting') "
        "                 || jsonb_build_object('show_sample_questions_in_greeting', "
        "                                       chat_theme->'show_sample_messages_in_greeting') "
        "WHERE chat_theme ? 'show_sample_messages_in_greeting'"
    ))
    op.alter_column("rag_models", "sample_messages", new_column_name="sample_questions")
    op.alter_column("messages", "response", new_column_name="answer")
    op.alter_column("messages", "message", new_column_name="question")
