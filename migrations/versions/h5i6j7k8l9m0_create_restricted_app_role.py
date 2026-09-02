"""Create the restricted application role and grant it what the app needs.

Split from the policy itself (the next revision) so that privileges and
enforcement roll back independently.

The role is created NOLOGIN and without a password — credentials are an operator
concern, not a schema one. To actually use it:

    ALTER ROLE ragr_app WITH LOGIN PASSWORD '<from the secret store>';

Nothing connects as this role yet. The API and worker still run as the owner, so
creating it is inert until a DATABASE_URL points at it.

Revision ID: h5i6j7k8l9m0
Revises: j7k8l9m0n1o2
Create Date: 2026-09-02
"""

from alembic import op

revision = "h5i6j7k8l9m0"
down_revision = "j7k8l9m0n1o2"
branch_labels = None
depends_on = None

APP_ROLE = "ragr_app"


def upgrade() -> None:
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                EXECUTE 'CREATE ROLE {APP_ROLE} NOLOGIN NOSUPERUSER '
                        'NOCREATEDB NOCREATEROLE NOBYPASSRLS';
            END IF;
        END $$;
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {APP_ROLE}")
    # Easy to forget, and its absence breaks every INSERT rather than any SELECT:
    # the id columns are SERIAL, so writes need the sequence too.
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")
    # Migrations are the owner's business.
    op.execute(f"REVOKE ALL ON TABLE alembic_version FROM {APP_ROLE}")
    # Default privileges are per-grantor: these only cover tables created by the
    # same role that runs this migration. A future migration run as a different
    # role would create tables the app cannot read.
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {APP_ROLE}"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE}"
    )


def downgrade() -> None:
    # The role is deliberately not dropped: roles are cluster-scoped, may hold
    # privileges in other databases, and dropping one out from under a running
    # deployment is worse than leaving an unused role behind. Drop it by hand.
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM {APP_ROLE}"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"REVOKE USAGE, SELECT ON SEQUENCES FROM {APP_ROLE}"
    )
    op.execute(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {APP_ROLE}")
    op.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {APP_ROLE}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {APP_ROLE}")
