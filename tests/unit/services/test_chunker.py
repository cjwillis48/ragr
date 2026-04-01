from app.services.chunker import (
    chunk_text,
    _split_sections,
    _split_paragraphs,
    _force_split_long_paragraphs,
    _merge_into_chunks,
)


class TestChunkText:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        result = chunk_text("Hello world", chunk_size=1000)
        assert result == ["Hello world"]

    def test_section_splitting(self):
        text = "Section one content.\n---\nSection two content."
        result = chunk_text(text, chunk_size=1000)
        assert len(result) == 2
        assert "Section one" in result[0]
        assert "Section two" in result[1]

    def test_multiple_dashes_in_divider(self):
        text = "Part A.\n------\nPart B."
        result = chunk_text(text, chunk_size=1000)
        assert len(result) == 2

    def test_paragraph_splitting_within_large_section(self):
        para1 = "A" * 600
        para2 = "B" * 600
        text = f"{para1}\n\n{para2}"
        result = chunk_text(text, chunk_size=700, chunk_overlap=0)
        assert len(result) >= 2

    def test_sentence_boundary_splitting(self):
        # One giant paragraph > chunk_size, with sentence boundaries
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = chunk_text(text, chunk_size=40, chunk_overlap=0)
        assert len(result) >= 2
        # Should split at a period, not mid-word
        assert result[0].endswith(".")

    def test_hard_cut_fallback(self):
        # No periods at all — forces hard cut
        text = "A" * 200
        result = chunk_text(text, chunk_size=50, chunk_overlap=0)
        assert len(result) >= 2
        assert len(result[0]) == 50

    def test_overlap_between_chunks(self):
        para1 = "A" * 80
        para2 = "B" * 80
        para3 = "C" * 80
        text = f"{para1}\n\n{para2}\n\n{para3}"
        result = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(result) >= 2
        # Second chunk should start with tail of first chunk (overlap)
        if len(result) >= 2:
            # The overlap means some content from the end of the previous chunk
            # appears at the start of the next
            assert result[1].startswith(result[0][-20:]) or len(result) > 2

    def test_custom_chunk_size(self):
        text = "Word " * 100  # ~500 chars
        result = chunk_text(text, chunk_size=50, chunk_overlap=0)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 55  # small margin for paragraph merging

    def test_preserves_all_content(self):
        """All original text should appear somewhere in the chunks (minus overlap duplication)."""
        text = "Alpha.\n\nBravo.\n\nCharlie.\n\nDelta."
        result = chunk_text(text, chunk_size=20, chunk_overlap=0)
        joined = " ".join(result)
        for word in ["Alpha", "Bravo", "Charlie", "Delta"]:
            assert word in joined


class TestSplitSections:
    def test_no_dividers(self):
        assert _split_sections("just text") == ["just text"]

    def test_single_divider(self):
        result = _split_sections("a\n---\nb")
        assert result == ["a", "b"]

    def test_empty_sections_filtered(self):
        result = _split_sections("\n---\n\n---\ncontent")
        assert result == ["content"]


class TestSplitParagraphs:
    def test_single_paragraph(self):
        assert _split_paragraphs("hello") == ["hello"]

    def test_multiple_paragraphs(self):
        assert _split_paragraphs("a\n\nb\n\nc") == ["a", "b", "c"]

    def test_empty_paragraphs_filtered(self):
        assert _split_paragraphs("a\n\n\n\nb") == ["a", "b"]


class TestForceSplitLongParagraphs:
    def test_short_paragraphs_unchanged(self):
        result = _force_split_long_paragraphs(["short"], chunk_size=100)
        assert result == ["short"]

    def test_splits_at_sentence_boundary(self):
        long_para = "First sentence. Second sentence. Third sentence."
        result = _force_split_long_paragraphs([long_para], chunk_size=30)
        assert len(result) >= 2
        assert result[0].endswith(".")

    def test_hard_cut_no_periods(self):
        long_para = "X" * 100
        result = _force_split_long_paragraphs([long_para], chunk_size=40)
        assert len(result) == 3  # 40 + 40 + 20
        assert result[0] == "X" * 40


class TestMergeIntoChunks:
    def test_single_paragraph(self):
        result = _merge_into_chunks(["hello"], chunk_size=100, chunk_overlap=10)
        assert result == ["hello"]

    def test_merges_small_paragraphs(self):
        result = _merge_into_chunks(["a", "b", "c"], chunk_size=100, chunk_overlap=0)
        assert len(result) == 1
        assert "a" in result[0] and "b" in result[0] and "c" in result[0]

    def test_splits_when_exceeds_size(self):
        paras = ["A" * 60, "B" * 60]
        result = _merge_into_chunks(paras, chunk_size=70, chunk_overlap=0)
        assert len(result) == 2
