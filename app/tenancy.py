"""Which tenant the current request or job is allowed to see.

The tenant is published to Postgres as the `app.model_id` setting, which the
row-level security policy on `content_chunks` reads. When it is unset the policy
matches nothing, so a query that forgets to scope by model returns no rows
instead of another tenant's.

This deliberately does not live in `app.middleware.log_context`, where the
contextvar started life: it is a security control now, and the database layer
must not depend on a logging module to know who the tenant is.

Two writers, because the tenant is not always known before the transaction is:

- `tenant_scope` — the tenant is known up front (the worker, tests). The
  `after_begin` listener in app.database picks it up when the transaction opens.
- `bind_tenant` — the tenant was discovered *by* a query, so the transaction is
  already open and the listener has already run without it. See its docstring.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_MODEL_ID: ContextVar[int | None] = ContextVar("current_model_id", default=None)

# Transaction-scoped (is_local => true), so it is discarded at COMMIT and at
# ROLLBACK — including ROLLBACK TO SAVEPOINT. The after_begin listener re-applies
# it on the next transaction, which is what keeps mid-request commits safe.
SET_TENANT = text("SELECT set_config('app.model_id', :model_id, true)")


def current_model_id() -> int | None:
    """The tenant bound to this context, or None if nothing has claimed one."""
    return _MODEL_ID.get()


@contextmanager
def tenant_scope(model_id: int | None) -> Iterator[None]:
    """Bind the tenant for the duration of the block, then restore what was there.

    Restoring is the point. Request handlers each get their own context, so a
    bare `set()` is contained there, but the worker processes every tenant's jobs
    from one long-lived task — a value left behind would silently become the next
    job's tenant.
    """
    token = _MODEL_ID.set(model_id)
    try:
        yield
    finally:
        _MODEL_ID.reset(token)


async def bind_tenant(session: AsyncSession, model_id: int) -> None:
    """Claim the tenant on a session whose transaction is already open.

    Resolving a slug to a model is itself a query, so by the time we know which
    tenant this request belongs to, the session has already autobegun and the
    `after_begin` listener has run with nothing to publish. Setting only the
    contextvar here would leave the GUC empty for the rest of the request, and
    every content_chunks read would come back empty — a silent, total retrieval
    failure that looks like a bad index rather than a misconfigured policy.

    So push it to the live transaction too. Forgetting to call this fails closed
    (no rows), never open.
    """
    _MODEL_ID.set(model_id)
    await session.execute(SET_TENANT, {"model_id": str(model_id)})
