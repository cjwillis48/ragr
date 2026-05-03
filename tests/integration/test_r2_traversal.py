"""Integration tests for R2 confirm-upload traversal protection.

The /sources/upload/confirm endpoint trusts the client-supplied object_key.
We must reject path traversal (which would otherwise let one tenant claim
another tenant's uploads) while still accepting legitimate filenames that
happen to contain consecutive dots (e.g. "report..final.pdf").
"""

import pytest

import app.api.sources as sources_module


@pytest.fixture(autouse=True)
def _enable_r2(monkeypatch):
    """Pretend R2 is configured so confirm_upload doesn't 501-out."""
    monkeypatch.setattr(sources_module, "r2_is_configured", lambda: True)


async def _create_model(client, slug: str = "r2-bot"):
    resp = await client.post("/models", json={"name": "R2 Bot", "slug": slug})
    assert resp.status_code == 201
    return resp.json()


class TestTraversalRejected:
    @pytest.mark.parametrize("evil_key", [
        "uploads/{model_id}/../other/file.pdf",
        "uploads/{model_id}/sub/../../other/file.pdf",
        "../uploads/{model_id}/file.pdf",
        "uploads/..",
    ])
    async def test_traversal_components_rejected(self, client, db_session, evil_key):
        model = await _create_model(client, "r2-trav-bot")
        from app.models.rag_model import RagModel
        from sqlalchemy import select
        result = await db_session.execute(select(RagModel).where(RagModel.slug == "r2-trav-bot"))
        model_id = result.scalar_one().id

        resp = await client.post("/models/r2-trav-bot/sources/upload/confirm", json={
            "upload_id": "x",
            "files": [{"filename": "f.pdf", "object_key": evil_key.format(model_id=model_id)}],
        })
        assert resp.status_code == 422
        assert "Invalid object key" in resp.json()["detail"]


class TestLegitimateFilenamesAccepted:
    async def test_double_dot_in_filename_not_rejected_as_traversal(self, client, db_session):
        """A filename like 'report..final.pdf' is fine — only path components equal to '..' are blocked."""
        model = await _create_model(client, "r2-legit-bot")
        from app.models.rag_model import RagModel
        from sqlalchemy import select
        result = await db_session.execute(select(RagModel).where(RagModel.slug == "r2-legit-bot"))
        model_id = result.scalar_one().id

        # This must NOT trigger the traversal 422.
        resp = await client.post("/models/r2-legit-bot/sources/upload/confirm", json={
            "upload_id": "y",
            "files": [{
                "filename": "report..final.pdf",
                "object_key": f"uploads/{model_id}/y/report..final.pdf",
            }],
        })
        # Endpoint should accept the request (202); 422 specifically would mean
        # the legitimate filename was wrongly rejected by the traversal check.
        assert resp.status_code != 422, resp.text
        assert resp.status_code == 202


class TestCrossTenantObjectKeyRejected:
    async def test_object_key_for_different_model_rejected(self, client, db_session):
        """Even without traversal, an object_key belonging to a different model must 403."""
        await _create_model(client, "r2-mine-bot")
        from app.models.rag_model import RagModel
        from sqlalchemy import select
        result = await db_session.execute(select(RagModel).where(RagModel.slug == "r2-mine-bot"))
        my_id = result.scalar_one().id

        # Use a plausible "other model" id — anything other than my_id.
        other_id = my_id + 9999

        resp = await client.post("/models/r2-mine-bot/sources/upload/confirm", json={
            "upload_id": "z",
            "files": [{"filename": "f.pdf", "object_key": f"uploads/{other_id}/z/f.pdf"}],
        })
        assert resp.status_code == 403
