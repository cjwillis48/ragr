"""Tenant context bookkeeping.

The database half of row-level security is covered by
tests/integration/test_rls_content_chunks.py. This is the half that decides which
tenant gets published in the first place — and the worker runs every tenant's jobs
from one long-lived task, so a value left behind becomes the next job's tenant.
"""

import pytest

from app.tenancy import current_model_id, tenant_scope


def test_no_tenant_by_default():
    assert current_model_id() is None


def test_scope_binds_and_restores():
    with tenant_scope(7):
        assert current_model_id() == 7
    assert current_model_id() is None


def test_scope_restores_on_exception():
    """A failing job must not leak its tenant into the next one."""
    with pytest.raises(RuntimeError):
        with tenant_scope(7):
            raise RuntimeError("job blew up")
    assert current_model_id() is None


def test_nested_scopes_restore_the_outer_tenant():
    with tenant_scope(1):
        with tenant_scope(2):
            assert current_model_id() == 2
        assert current_model_id() == 1
    assert current_model_id() is None


def test_scope_can_clear_an_inherited_tenant():
    """Request boundaries clear inherited context rather than trusting it."""
    with tenant_scope(1):
        with tenant_scope(None):
            assert current_model_id() is None
        assert current_model_id() == 1
