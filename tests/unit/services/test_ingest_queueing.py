"""Queueing a Source for ingestion.

These functions exist because five call sites used to assemble the same two
writes by hand: upsert the Source, then enqueue the Job. The pairing is the
invariant worth protecting — a Job with no Source row has nothing to report
failure against — so the tests check the pair, not the halves.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.ingestion_job import IngestionJob
from app.services.ingest import queue_crawl, queue_file, queue_r2_file, queue_url


@pytest.fixture
def session():
    s = MagicMock()
    s.execute = AsyncMock()
    s.add = MagicMock()
    return s


def _job(session) -> IngestionJob:
    """The Job handed to session.add(), failing loudly if none was."""
    assert session.add.call_count == 1, "expected exactly one Job to be enqueued"
    job = session.add.call_args[0][0]
    assert isinstance(job, IngestionJob)
    return job


def _source_values(session) -> dict:
    """The values of the Source upsert, read off the compiled statement."""
    assert session.execute.await_count == 1, "expected exactly one Source upsert"
    stmt = session.execute.await_args[0][0]
    return stmt.compile().params


class TestPairing:
    """Every queue_* writes a Source and a Job, scoped to the same Model."""

    async def test_url_writes_both(self, session, sample_model):
        await queue_url(session, sample_model, "doc.md", "https://example.com/a")
        assert _source_values(session)["source_identifier"] == "doc.md"
        assert _job(session).model_id == sample_model.id

    async def test_file_writes_both(self, session, sample_model):
        await queue_file(session, sample_model, "notes.md", "text", "hello")
        assert _source_values(session)["raw_content"] == "hello"
        assert _job(session).job_type == "file"

    async def test_r2_writes_both(self, session, sample_model):
        await queue_r2_file(session, sample_model, "a.pdf", "uploads/a.pdf")
        assert _job(session).job_params == {"object_key": "uploads/a.pdf", "filename": "a.pdf"}

    async def test_crawl_writes_both(self, session, sample_model):
        await queue_crawl(
            session, sample_model, "https://example.com",
            max_pages=10, max_depth=2, prefix=None, exclude_patterns=None,
        )
        assert _job(session).job_type == "crawl"


class TestSourceShape:
    async def test_crawl_root_is_crawling_not_pending(self, session, sample_model):
        """The root Source stands in for the crawl, so it isn't waiting to be ingested."""
        await queue_crawl(
            session, sample_model, "https://example.com",
            max_pages=1, max_depth=1, prefix=None, exclude_patterns=None,
        )
        assert _source_values(session)["status"] == "crawling"

    async def test_r2_content_type_deferred(self, session, sample_model):
        """The worker decides the type once it has the bytes."""
        await queue_r2_file(session, sample_model, "a.pdf", "uploads/a.pdf")
        assert _source_values(session)["content_type"] == "pending"

    async def test_upsert_preserves_last_completed_ingest(self, session, sample_model):
        """content_hash and chunk_count describe the last finished ingest.

        Overwriting them here would make a Source that fails re-ingestion look
        like it has no content at all.
        """
        await queue_url(session, sample_model, "doc.md", "https://example.com/a")
        stmt = session.execute.await_args[0][0]
        updated_cols = {col for col, _ in stmt._post_values_clause.update_values_to_set}
        assert "status" in updated_cols
        assert "content_hash" not in updated_cols
        assert "chunk_count" not in updated_cols


class TestJobParams:
    async def test_crawled_page_carries_its_parent(self, session, sample_model):
        """A page found by a crawl is traceable back to the crawl that found it."""
        await queue_file(
            session, sample_model, "https://example.com/p", "html", "text",
            source_url="https://example.com/p", parent_job_id=42,
        )
        assert _job(session).job_params["parent_job_id"] == 42

    async def test_parent_omitted_when_absent(self, session, sample_model):
        await queue_file(session, sample_model, "notes.md", "text", "hello")
        assert "parent_job_id" not in _job(session).job_params
