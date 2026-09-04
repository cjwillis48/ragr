"""Make the database enforce tenant isolation on content_chunks.

Application code has always scoped chunk queries by model_id. This makes a query
that forgets the filter return nothing rather than another tenant's rows —
turning a silent leak into a visibly broken feature.

`app/database.py` publishes the tenant as `app.model_id` at the start of every
transaction; `app/tenancy.py` explains the two places that set it.

Currently INERT in every environment: the app connects as the initdb superuser,
and superusers bypass row security unconditionally — FORCE does not change that.
FORCE is included so protection does not silently disappear the day the owner
role is de-superusered. Note that if that ever happens, data migrations touching
content_chunks will start matching zero rows unless they set the GUC first.

Revision ID: i6j7k8l9m0n1
Revises: h5i6j7k8l9m0
Create Date: 2026-09-02
"""

from alembic import op

revision = "i6j7k8l9m0n1"
down_revision = "h5i6j7k8l9m0"
branch_labels = None
depends_on = None

# NULLIF maps our explicit "no tenant" sentinel onto NULL, and `model_id = NULL`
# is NULL rather than true — so an unscoped query filters everything out. The
# `true` second argument to current_setting means a never-set GUC returns NULL
# instead of raising, so this fails closed rather than 500ing.
_PREDICATE = "model_id = NULLIF(current_setting('app.model_id', true), '')::integer"


def upgrade() -> None:
    op.execute("ALTER TABLE content_chunks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE content_chunks FORCE ROW LEVEL SECURITY")
    # TO PUBLIC rather than naming the app role: an unexpected role should be
    # denied by default, not exempted by omission.
    op.execute(
        f"""
        CREATE POLICY content_chunks_tenant_isolation ON content_chunks
            FOR ALL
            TO PUBLIC
            USING ({_PREDICATE})
            WITH CHECK ({_PREDICATE})
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS content_chunks_tenant_isolation ON content_chunks")
    op.execute("ALTER TABLE content_chunks NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE content_chunks DISABLE ROW LEVEL SECURITY")
