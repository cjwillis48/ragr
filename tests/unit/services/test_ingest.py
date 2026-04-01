import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ingest import ingest_content, IngestResult
from app.services.embedder import EmbedResult


class TestIngestContent:
    def _mock_session(self, existing_source=None):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=existing_source)
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    def _mock_embed(self, n_chunks: int):
        return EmbedResult(
            embeddings=[[0.1, 0.2] for _ in range(n_chunks)],
            total_tokens=n_chunks * 100,
        )

    async def test_new_content_ingested(self, sample_model):
        session = self._mock_session(existing_source=None)

        with (
            patch("app.services.ingest.chunk_text", return_value=["chunk1", "chunk2"]),
            patch("app.services.ingest.embed_texts", new_callable=AsyncMock, return_value=self._mock_embed(2)),
            patch("app.services.ingest.estimate_embedding_cost", return_value=0.001),
        ):
            result = await ingest_content(session, sample_model, "hello world", "source-1")

        assert result.chunk_count == 2
        assert result.skipped is False
        assert result.embedding_cost == 0.001
        assert session.add.call_count == 2  # two chunks added
        session.commit.assert_called_once()

    async def test_idempotent_same_hash(self, sample_model):
        """Same content hash → skipped."""
        import hashlib

        content = "test content"
        h = f"{content}:chunk_size={sample_model.chunk_size}:chunk_overlap={sample_model.chunk_overlap}"
        expected_hash = hashlib.sha256(h.encode()).hexdigest()

        existing = MagicMock()
        existing.content_hash = expected_hash
        existing.chunk_count = 5
        existing.status = "complete"

        session = self._mock_session(existing_source=existing)

        result = await ingest_content(session, sample_model, content, "source-1")

        assert result.skipped is True
        assert result.chunk_count == 5
        assert result.embedding_cost == 0.0

    async def test_changed_content_reingests(self, sample_model):
        """Different hash → old chunks deleted, new ones created."""
        existing = MagicMock()
        existing.content_hash = "old-hash"
        existing.status = "complete"

        session = self._mock_session(existing_source=existing)

        with (
            patch("app.services.ingest.chunk_text", return_value=["new-chunk"]),
            patch("app.services.ingest.embed_texts", new_callable=AsyncMock, return_value=self._mock_embed(1)),
            patch("app.services.ingest.estimate_embedding_cost", return_value=0.0005),
        ):
            result = await ingest_content(session, sample_model, "new content", "source-1")

        assert result.skipped is False
        assert result.chunk_count == 1
        # session.execute called: once for existing check, once for delete, once for upsert
        assert session.execute.call_count >= 3

    async def test_empty_content_returns_zero(self, sample_model):
        session = self._mock_session(existing_source=None)

        with patch("app.services.ingest.chunk_text", return_value=[]):
            result = await ingest_content(session, sample_model, "   ", "source-1")

        assert result.chunk_count == 0
        assert result.skipped is False

    async def test_rollback_on_error(self, sample_model):
        session = self._mock_session(existing_source=None)

        with (
            patch("app.services.ingest.chunk_text", return_value=["chunk"]),
            patch("app.services.ingest.embed_texts", new_callable=AsyncMock, side_effect=RuntimeError("API error")),
            pytest.raises(RuntimeError, match="API error"),
        ):
            await ingest_content(session, sample_model, "content", "source-1")

        session.rollback.assert_called_once()

    async def test_incomplete_source_marked_complete(self, sample_model):
        """Existing source with status != 'complete' but matching hash gets updated."""
        import hashlib

        content = "test"
        h = f"{content}:chunk_size={sample_model.chunk_size}:chunk_overlap={sample_model.chunk_overlap}"
        expected_hash = hashlib.sha256(h.encode()).hexdigest()

        existing = MagicMock()
        existing.content_hash = expected_hash
        existing.chunk_count = 3
        existing.status = "pending"

        session = self._mock_session(existing_source=existing)

        result = await ingest_content(session, sample_model, content, "source-1")

        assert result.skipped is True
        assert existing.status == "complete"
        session.commit.assert_called_once()
