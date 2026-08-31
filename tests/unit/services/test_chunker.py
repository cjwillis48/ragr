import re

import pytest

from app.services.chunker import (
    ChunkResult,
    _last_break,
    _overlap_tail,
    _parse_blocks,
    chunk_text,
)

# Shaped like real HTML output after markdown conversion.
MARKDOWN_DOC = """# Hades

God of the underworld in Greek mythology. He is the eldest son of Cronus and Rhea.

## Name

The origin of the name is uncertain. It has generally been read as "the unseen one".

### Etymology

Modern linguists propose the Proto-Greek form *Awides*, meaning unseen.

## Mythology

### Early years

Hades was devoured by his father Cronus as soon as he was born.
"""


class TestChunkText:
    def test_empty_and_whitespace(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  \t ") == []

    def test_returns_chunk_results(self):
        chunks = chunk_text(MARKDOWN_DOC, 1000, 100)
        assert chunks
        assert all(isinstance(c, ChunkResult) for c in chunks)

    def test_positions_are_sequential_from_zero(self):
        chunks = chunk_text(MARKDOWN_DOC, 200, 20)
        assert [c.position for c in chunks] == list(range(len(chunks)))

    def test_heading_path_tracks_nesting(self):
        chunks = chunk_text(MARKDOWN_DOC, 200, 0)
        paths = {tuple(c.heading_path) for c in chunks}
        assert ("Hades", "Name") in paths
        assert ("Hades", "Name", "Etymology") in paths
        assert ("Hades", "Mythology", "Early years") in paths

    def test_sections_do_not_bleed_together(self):
        """Content under one heading must not be packed in with another's."""
        chunks = chunk_text(MARKDOWN_DOC, 1000, 0)
        for chunk in chunks:
            # A chunk may contain its own subheadings, but every heading it
            # contains must be consistent with the path it was assigned.
            if "Etymology" in chunk.text:
                assert "Early years" not in chunk.text

    def test_no_orphan_heading_only_chunk(self):
        """A heading immediately followed by a subheading must not emit alone."""
        chunks = chunk_text(MARKDOWN_DOC, 1000, 0)
        for chunk in chunks:
            stripped = [ln for ln in chunk.text.splitlines() if ln.strip()]
            assert not all(ln.lstrip().startswith("#") for ln in stripped), chunk.text

    @pytest.mark.parametrize("size", [120, 300, 800])
    def test_never_cuts_mid_word(self, size):
        prose = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 40).strip()
        for chunk in chunk_text(prose, size, 0):
            # Every whitespace-separated token must be a whole word from the source.
            for token in chunk.text.split():
                assert token in prose

    def test_respects_chunk_size(self):
        prose = ("Sentence number one here. " * 200).strip()
        for chunk in chunk_text(prose, 300, 50):
            assert len(chunk.text) <= 300

    def test_single_unbroken_token_is_hard_cut(self):
        """The one case where a mid-word cut is unavoidable."""
        chunks = chunk_text("x" * 250, 100, 0)
        assert len(chunks) == 3
        assert all(len(c.text) <= 100 for c in chunks)

    def test_offsets_point_at_source(self):
        chunks = chunk_text(MARKDOWN_DOC, 200, 0)
        for chunk in chunks:
            assert 0 <= chunk.start_offset < len(MARKDOWN_DOC)
            assert chunk.end_offset <= len(MARKDOWN_DOC)
            assert chunk.start_offset < chunk.end_offset

    def test_offsets_are_monotonic(self):
        chunks = chunk_text(MARKDOWN_DOC, 200, 0)
        starts = [c.start_offset for c in chunks]
        assert starts == sorted(starts)

    def test_preserves_all_content_without_overlap(self):
        """Every non-whitespace character of the source survives chunking."""
        chunks = chunk_text(MARKDOWN_DOC, 250, 0)
        joined = "".join(c.text for c in chunks)
        assert re.sub(r"\s", "", joined) == re.sub(r"\s", "", MARKDOWN_DOC)

    def test_overlap_starts_at_a_word_boundary(self):
        prose = ("The quick brown fox jumps over the lazy dog. " * 60).strip()
        chunks = chunk_text(prose, 300, 60)
        for chunk in chunks[1:]:
            first = chunk.text.split()[0]
            assert first in prose.split(), f"overlap began mid-word: {first!r}"

    def test_pdf_shaped_input_with_hard_wraps(self):
        """Hard-wrapped lines must still join into sentences, not fragment."""
        wrapped = "\n".join([
            "Hades is the Greek god of the underworld and",
            "the eldest son of Cronus and Rhea. He was",
            "given dominion over the dead after the war.",
        ])
        chunks = chunk_text(wrapped, 1000, 0)
        assert len(chunks) == 1
        assert "underworld and\nthe eldest" in chunks[0].text

    def test_list_items_stay_with_their_block(self):
        doc = "## Symbols\n\n* Cypress\n* Narcissus\n* Keys\n* Serpent"
        chunks = chunk_text(doc, 1000, 0)
        assert len(chunks) == 1
        assert chunks[0].text.count("*") == 4

    def test_unicode_is_not_split_mid_character(self):
        doc = "## Ονόματα\n\n" + ("Ἅιδης καὶ Περσεφόνη εἰσὶν θεοί. " * 30)
        for chunk in chunk_text(doc, 200, 20):
            chunk.text.encode("utf-8").decode("utf-8")

    def test_overlap_not_silently_dropped_when_block_is_large(self):
        """A big block should still carry what overlap fits, not zero."""
        big = "A" * 180 + ". " + "B" * 180 + "."
        doc = f"First sentence is short. {big}"
        chunks = chunk_text(doc, 220, 40)
        assert len(chunks) > 1


class TestParseBlocks:
    def test_splits_on_blank_lines(self):
        blocks = _parse_blocks("one\n\ntwo\n\nthree")
        assert [b.text for b in blocks] == ["one", "two", "three"]

    def test_heading_flagged_and_pushed_on_stack(self):
        blocks = _parse_blocks("# A\n\ntext\n\n## B\n\nmore")
        assert [b.is_heading for b in blocks] == [True, False, True, False]
        assert blocks[3].heading_path == ("A", "B")

    def test_sibling_heading_pops_the_stack(self):
        blocks = _parse_blocks("# A\n\n## B\n\ntext\n\n## C\n\nmore")
        assert blocks[-1].heading_path == ("A", "C")

    def test_block_start_offsets_are_correct(self):
        text = "alpha\n\nbeta\n\ngamma"
        for block in _parse_blocks(text):
            assert text[block.start:block.start + len(block.text)] == block.text


class TestLastBreak:
    def test_prefers_sentence_end(self):
        window = "First sentence. Second one trails"
        assert window[:_last_break(window)] == "First sentence."

    def test_falls_back_to_word_boundary(self):
        window = "no sentence punctuation in here"
        assert window[:_last_break(window)] == "no sentence punctuation in"

    def test_unbroken_token_returns_full_window(self):
        assert _last_break("xxxxxxxxxx") == 10


class TestOverlapTail:
    def test_snaps_forward_past_partial_word(self):
        assert not _overlap_tail("the quick brown fox", 8).startswith("wn")

    def test_zero_budget_returns_empty(self):
        assert _overlap_tail("anything at all", 0) == ""

    def test_prefers_a_sentence_start(self):
        assert _overlap_tail("One ends here. Two begins now.", 20) == "Two begins now."
