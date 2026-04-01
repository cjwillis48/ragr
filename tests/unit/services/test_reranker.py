import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import app.services.reranker as reranker_module
from app.services.reranker import rerank, RerankResult


@pytest.fixture(autouse=True)
def reset_reranker_globals():
    reranker_module._platform_client = None
    reranker_module._client_cache.clear()
    yield
    reranker_module._platform_client = None
    reranker_module._client_cache.clear()


def _mock_rerank_response(indices, scores, tokens=100):
    result = MagicMock()
    result.results = [
        MagicMock(index=idx, relevance_score=score) for idx, score in zip(indices, scores)
    ]
    result.total_tokens = tokens
    return result


class TestRerank:
    async def test_returns_rerank_result(self):
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(
            return_value=_mock_rerank_response([2, 0, 1], [0.95, 0.80, 0.60], 150)
        )

        with patch.object(reranker_module, "_get_client", return_value=mock_client):
            result = await rerank("query", ["doc0", "doc1", "doc2"], top_k=3)

        assert isinstance(result, RerankResult)
        assert result.indices == [2, 0, 1]
        assert result.scores == [0.95, 0.80, 0.60]
        assert result.total_tokens == 150

    async def test_top_k_passed_through(self):
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(
            return_value=_mock_rerank_response([0], [0.9], 50)
        )

        with patch.object(reranker_module, "_get_client", return_value=mock_client):
            await rerank("query", ["doc0", "doc1"], top_k=1)

        _, kwargs = mock_client.rerank.call_args
        assert kwargs["top_k"] == 1

    async def test_model_passed_through(self):
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(
            return_value=_mock_rerank_response([0], [0.9], 50)
        )

        with patch.object(reranker_module, "_get_client", return_value=mock_client):
            await rerank("query", ["doc"], model="rerank-2", top_k=1)

        _, kwargs = mock_client.rerank.call_args
        assert kwargs["model"] == "rerank-2"
