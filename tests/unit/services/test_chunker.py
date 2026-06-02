from app.services.chunker import (
    chunk_code,
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


class TestChunkCode:
    def test_empty_string(self):
        assert chunk_code("") == []

    def test_whitespace_only(self):
        assert chunk_code("   \n\n  ") == []

    def test_single_line(self):
        result = chunk_code("print('hi')", chunk_size=1000)
        assert result == [("print('hi')", 1, 1)]

    def test_single_line_with_trailing_newline(self):
        # splitlines() strips trailing newline — last line is still line 1
        result = chunk_code("print('hi')\n", chunk_size=1000)
        assert result == [("print('hi')", 1, 1)]

    def test_no_newlines_short(self):
        # Single line, no newline at all — still works
        result = chunk_code("a" * 50, chunk_size=1000)
        assert len(result) == 1
        assert result[0][1] == 1
        assert result[0][2] == 1

    def test_short_file_fits_in_one_chunk(self):
        text = "line1\nline2\nline3"
        result = chunk_code(text, chunk_size=1000)
        assert len(result) == 1
        chunk, start, end = result[0]
        assert chunk == "line1\nline2\nline3"
        assert start == 1
        assert end == 3

    def test_multi_chunk_line_ranges_are_inclusive(self):
        # Each "line" is 20 chars. chunk_size=50 fits about 2 lines per chunk.
        lines = [f"line_{i:02d}_{'x'*12}" for i in range(6)]
        text = "\n".join(lines)
        result = chunk_code(text, chunk_size=50, chunk_overlap=0)
        # Every chunk should have a valid start/end and they should cover all 6 lines.
        assert len(result) >= 2
        first_start = result[0][1]
        last_end = result[-1][2]
        assert first_start == 1
        assert last_end == 6
        for chunk, start, end in result:
            assert 1 <= start <= end <= 6
            # The chunk content should match the source lines joined by \n
            assert chunk == "\n".join(lines[start - 1:end])

    def test_overlap_repeats_trailing_lines(self):
        lines = [f"line_{i}_{'x'*15}" for i in range(6)]  # ~22 chars each
        text = "\n".join(lines)
        # chunk_size=50 → ~2 lines per chunk; overlap=25 → one trailing line repeats
        result = chunk_code(text, chunk_size=50, chunk_overlap=25)
        assert len(result) >= 2
        # Adjacent chunks should overlap on at least one line
        for i in range(len(result) - 1):
            _, _, end_prev = result[i]
            _, start_next, _ = result[i + 1]
            assert start_next <= end_prev, (
                f"Expected overlap: chunk {i} ends at {end_prev}, chunk {i+1} starts at {start_next}"
            )

    def test_line_longer_than_chunk_size_is_its_own_chunk(self):
        # A single line bigger than chunk_size must still produce a chunk (no infinite loop)
        big_line = "x" * 200
        text = f"short\n{big_line}\nshort2"
        result = chunk_code(text, chunk_size=50, chunk_overlap=0)
        # The big line must appear as its own chunk
        chunks_containing_big = [r for r in result if big_line in r[0]]
        assert len(chunks_containing_big) >= 1
        # And it should be tagged as line 2 (1-indexed)
        big_chunk = chunks_containing_big[0]
        assert big_chunk[1] == 2 and big_chunk[2] == 2

    def test_blank_lines_preserved_in_count(self):
        # Blank lines count as real lines for the start/end tracking
        text = "first\n\n\nfourth"
        result = chunk_code(text, chunk_size=1000)
        chunk, start, end = result[0]
        assert chunk == "first\n\n\nfourth"
        assert start == 1
        assert end == 4

    def test_no_overlap(self):
        lines = ["aaa", "bbb", "ccc", "ddd"]
        text = "\n".join(lines)
        # chunk_size=8 fits 2 lines (3 + 1 + 3 = 7)
        result = chunk_code(text, chunk_size=8, chunk_overlap=0)
        # No overlap → each line appears exactly once across chunks (in order)
        all_lines_collected = []
        for chunk, _, _ in result:
            all_lines_collected.extend(chunk.split("\n"))
        assert all_lines_collected == lines

    def test_progresses_even_when_overlap_would_repeat_everything(self):
        # If chunk_overlap is large enough that it would repeat the whole previous chunk,
        # we still advance by at least one line.
        lines = ["aa", "bb", "cc"]
        text = "\n".join(lines)
        result = chunk_code(text, chunk_size=5, chunk_overlap=10_000)
        # Must terminate and cover all lines
        end_lines = {r[2] for r in result}
        assert 3 in end_lines  # last line is reached
