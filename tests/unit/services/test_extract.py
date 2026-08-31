import pytest

from app.services.extract import ExtractionError, extract_text


def _pdf_bytes(lines: list[str]) -> bytes:
    """A minimal single-page PDF containing the given lines of text."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "\n".join(lines), fontsize=11)
    return doc.tobytes()


class TestExtractText:
    def test_rejects_unsupported_extension(self):
        with pytest.raises(ExtractionError, match="Unsupported file type"):
            extract_text("virus.exe", b"whatever")

    def test_rejects_non_utf8(self):
        with pytest.raises(ExtractionError, match="UTF-8"):
            extract_text("notes.txt", b"\xff\xfe\x00bad")

    def test_plain_text_passes_through(self):
        text, content_type = extract_text("notes.txt", b"hello world")
        assert text == "hello world"
        assert content_type == "text"

    def test_markdown_preserved_verbatim(self):
        raw = b"# Title\n\nBody text here."
        text, content_type = extract_text("doc.md", raw)
        assert text == raw.decode()
        assert content_type == "markdown"

    def test_html_converted_to_markdown(self):
        raw = b'<body><h2>Pricing</h2><p>Costs $10 per <a href="/x">month</a>.</p></body>'
        text, content_type = extract_text("page.html", raw)
        assert content_type == "html"
        assert "## Pricing" in text
        assert "Costs $10 per month." in text

    def test_csv_and_json_treated_as_text(self):
        assert extract_text("data.csv", b"a,b\n1,2")[1] == "text"
        assert extract_text("data.json", b'{"a": 1}')[1] == "text"

    @pytest.mark.slow
    def test_pdf_lines_are_dewrapped_into_sentences(self):
        """page.get_text() keeps visual wrapping; to_markdown joins the sentence."""
        raw = _pdf_bytes([
            "Hades is the Greek god of the underworld and the",
            "eldest son of Cronus and Rhea. He received dominion",
            "over the dead after the war against the Titans.",
        ] + ["Filler line to clear the scanned-PDF guard."] * 4)
        text, content_type = extract_text("doc.pdf", raw)
        assert content_type == "pdf"
        assert "Hades is the Greek god" in text

    @pytest.mark.slow
    def test_scanned_pdf_rejected(self):
        with pytest.raises(ExtractionError, match="scanned"):
            extract_text("scan.pdf", _pdf_bytes(["hi"]))

    def test_unparseable_pdf_rejected(self):
        with pytest.raises(ExtractionError, match="Could not parse PDF"):
            extract_text("broken.pdf", b"this is not a pdf at all")
