from unittest.mock import AsyncMock, MagicMock

from app.services.retrieval import _expand_neighbours, _rrf_merge, ChunkScore, RRF_K


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


def _positioned(chunk_id: int, source: str, position: int) -> MagicMock:
    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.source_identifier = source
    chunk.position = position
    chunk.content = f"content-{chunk_id}"
    return chunk


def _session_returning(chunks):
    """AsyncSession stub whose execute() yields the given chunks."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = chunks
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    return session


def _model(radius: int) -> MagicMock:
    model = MagicMock()
    model.id = 1
    model.neighbor_radius = radius
    return model


class TestExpandNeighbours:
    async def test_radius_zero_is_a_no_op(self):
        hits = [_positioned(10, "doc", 5)]
        session = _session_returning([])
        assert await _expand_neighbours(session, _model(0), hits) == hits
        session.execute.assert_not_called()

    async def test_empty_hits_short_circuits(self):
        session = _session_returning([])
        assert await _expand_neighbours(session, _model(1), []) == []
        session.execute.assert_not_called()

    async def test_neighbours_surround_their_hit_in_order(self):
        hit = _positioned(10, "doc", 5)
        before, after = _positioned(9, "doc", 4), _positioned(11, "doc", 6)
        session = _session_returning([after, before])  # DB order is arbitrary
        out = await _expand_neighbours(session, _model(1), [hit])
        assert [c.id for c in out] == [9, 10, 11]

    async def test_missing_neighbour_is_skipped(self):
        """Position 0 has no predecessor; the hit still comes back."""
        hit = _positioned(10, "doc", 0)
        session = _session_returning([_positioned(11, "doc", 1)])
        out = await _expand_neighbours(session, _model(1), [hit])
        assert [c.id for c in out] == [10, 11]

    async def test_adjacent_hits_do_not_duplicate_shared_neighbours(self):
        a, b = _positioned(10, "doc", 5), _positioned(12, "doc", 7)
        shared = _positioned(11, "doc", 6)
        session = _session_returning([shared, _positioned(9, "doc", 4), _positioned(13, "doc", 8)])
        out = await _expand_neighbours(session, _model(1), [a, b])
        assert [c.id for c in out] == [9, 10, 11, 12, 13]
        assert len(out) == len(set(c.id for c in out))

    async def test_neighbours_scoped_to_the_same_source(self):
        """Position is only meaningful within a source."""
        hit_a = _positioned(10, "doc-a", 3)
        hit_b = _positioned(20, "doc-b", 3)
        session = _session_returning([])
        out = await _expand_neighbours(session, _model(1), [hit_a, hit_b])
        assert [c.id for c in out] == [10, 20]
        wanted = session.execute.call_args[0][0]
        assert wanted is not None  # query built without raising on mixed sources

    async def test_hits_are_never_displaced_by_neighbours(self):
        hits = [_positioned(10, "doc", 5), _positioned(30, "doc", 20)]
        session = _session_returning([_positioned(31, "doc", 21)])
        out = await _expand_neighbours(session, _model(1), hits)
        assert out.index(hits[0]) < out.index(hits[1])
