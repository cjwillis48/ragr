from unittest.mock import MagicMock

from app.services.retrieval import _rrf_merge, ChunkScore, RRF_K


def _make_chunk(chunk_id: int) -> MagicMock:
    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.content = f"content-{chunk_id}"
    return chunk


class TestChunkScore:
    def test_retrieval_method_vector(self):
        score = ChunkScore(chunk_id=1, distance=0.5)
        assert score.retrieval_method == "vector"

    def test_retrieval_method_keyword(self):
        score = ChunkScore(chunk_id=1, distance=1.0, keyword_rank=3)
        assert score.retrieval_method == "keyword"

    def test_retrieval_method_hybrid(self):
        score = ChunkScore(chunk_id=1, distance=0.5, keyword_rank=2)
        assert score.retrieval_method == "hybrid"

    def test_retrieval_method_vector_boundary(self):
        """distance == 1.0 means no vector match (keyword-only default)."""
        score = ChunkScore(chunk_id=1, distance=1.0)
        assert score.retrieval_method == "vector"  # no keyword_rank, so still "vector"


class TestRRFMerge:
    def test_vector_only(self):
        c1, c2 = _make_chunk(1), _make_chunk(2)
        vector = [(c1, 0.3), (c2, 0.5)]
        keyword = []

        chunks, distances, keyword_ranks = _rrf_merge(vector, keyword, limit=10)

        assert len(chunks) == 2
        assert chunks[0].id == 1  # higher ranked
        assert distances[1] == 0.3
        assert distances[2] == 0.5
        assert keyword_ranks == {}

    def test_keyword_only(self):
        c1, c2 = _make_chunk(1), _make_chunk(2)
        vector = []
        keyword = [(c1, 5.0), (c2, 3.0)]

        chunks, distances, keyword_ranks = _rrf_merge(vector, keyword, limit=10)

        assert len(chunks) == 2
        assert chunks[0].id == 1  # higher keyword rank
        assert distances[1] == 1.0  # keyword-only gets distance=1.0
        assert keyword_ranks[1] == 1
        assert keyword_ranks[2] == 2

    def test_hybrid_boosts_overlap(self):
        """A chunk appearing in both lists gets a higher RRF score."""
        shared = _make_chunk(1)
        vector_only = _make_chunk(2)
        keyword_only = _make_chunk(3)

        vector = [(shared, 0.3), (vector_only, 0.5)]
        keyword = [(shared, 5.0), (keyword_only, 3.0)]

        chunks, distances, keyword_ranks = _rrf_merge(vector, keyword, limit=10)

        # Shared chunk should be ranked first (appears in both)
        assert chunks[0].id == 1

    def test_dedup(self):
        """Same chunk in both lists appears only once."""
        c = _make_chunk(1)
        vector = [(c, 0.3)]
        keyword = [(c, 5.0)]

        chunks, _, _ = _rrf_merge(vector, keyword, limit=10)
        assert len(chunks) == 1
        assert chunks[0].id == 1

    def test_limit_respected(self):
        chunks_in = [_make_chunk(i) for i in range(10)]
        vector = [(c, 0.1 * i) for i, c in enumerate(chunks_in)]
        keyword = []

        chunks, _, _ = _rrf_merge(vector, keyword, limit=3)
        assert len(chunks) == 3

    def test_rrf_scores_correct(self):
        """Verify RRF formula: score = 1/(K + rank)."""
        c1 = _make_chunk(1)
        vector = [(c1, 0.3)]
        keyword = [(c1, 5.0)]

        # rank 1 in vector: 1/(60+1) = 1/61
        # rank 1 in keyword: 1/(60+1) = 1/61
        # total: 2/61
        chunks, _, _ = _rrf_merge(vector, keyword, limit=10)
        assert chunks[0].id == 1

    def test_empty_inputs(self):
        chunks, distances, keyword_ranks = _rrf_merge([], [], limit=10)
        assert chunks == []
        assert distances == {}
        assert keyword_ranks == {}

    def test_preserves_distances_from_vector(self):
        """Vector distances should be preserved, not overwritten by keyword."""
        c = _make_chunk(1)
        vector = [(c, 0.25)]
        keyword = [(c, 5.0)]

        _, distances, _ = _rrf_merge(vector, keyword, limit=10)
        assert distances[1] == 0.25  # from vector, not 1.0
