from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.services.embedder as embedder_module
from app.services.embedder import embed_texts, embed_query


@pytest.fixture(autouse=True)
def reset_embedder_globals():
    embedder_module._clients._platform_client = None
    embedder_module._clients._cache.clear()
    yield
    embedder_module._clients._platform_client = None
    embedder_module._clients._cache.clear()


def _mock_embed_response(n_texts: int, tokens: int = 100):
    result = MagicMock()
    result.embeddings = [[0.1, 0.2, 0.3] for _ in range(n_texts)]
    result.total_tokens = tokens
    return result


class TestEmbedTexts:
    async def test_empty_list(self):
        result = await embed_texts([])
        assert result.embeddings == []
        assert result.total_tokens == 0

    async def test_single_batch(self):
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(3, 300))

        with patch.object(embedder_module, "_get_client", return_value=mock_client):
            result = await embed_texts(["a", "b", "c"], batch_size=128)

        assert len(result.embeddings) == 3
        assert result.total_tokens == 300
        mock_client.embed.assert_called_once()

    async def test_multiple_batches(self):
        mock_client = AsyncMock()
        # Two batches: 2 texts each
        mock_client.embed = AsyncMock(
            side_effect=[
                _mock_embed_response(2, 200),
                _mock_embed_response(2, 200),
            ]
        )

        with patch.object(embedder_module, "_get_client", return_value=mock_client):
            result = await embed_texts(["a", "b", "c", "d"], batch_size=2)

        assert len(result.embeddings) == 4
        assert result.total_tokens == 400
        assert mock_client.embed.call_count == 2

    async def test_document_input_type(self):
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(1, 50))

        with patch.object(embedder_module, "_get_client", return_value=mock_client):
            await embed_texts(["text"])

        _, kwargs = mock_client.embed.call_args
        assert kwargs["input_type"] == "document"


class TestEmbedQuery:
    async def test_returns_single_embedding(self):
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(1, 50))

        with patch.object(embedder_module, "_get_client", return_value=mock_client):
            result = await embed_query("what is X?")

        assert result == [0.1, 0.2, 0.3]

    async def test_query_input_type(self):
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(1, 50))

        with patch.object(embedder_module, "_get_client", return_value=mock_client):
            await embed_query("what is X?")

        _, kwargs = mock_client.embed.call_args
        assert kwargs["input_type"] == "query"
