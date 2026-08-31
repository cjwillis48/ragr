"""Text extraction from uploaded file bytes, shared by the API and the worker."""

import logging
from pathlib import Path

import pymupdf
import pymupdf4llm

from app.services.html import strip_html

logger = logging.getLogger("ragr.extract")

ALLOWED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf", ".csv", ".json"}

# Below this, a PDF is almost certainly scanned images with no text layer.
_MIN_PDF_CHARS = 100


class ExtractionError(Exception):
    """Raised when a file cannot be turned into usable text."""


def extract_text(filename: str, raw: bytes) -> tuple[str, str]:
    """Extract text and content_type from raw file bytes. Raises ExtractionError on failure."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ExtractionError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        try:
            doc = pymupdf.Document(stream=raw, filetype="pdf")
        except Exception:
            raise ExtractionError("Could not parse PDF")
        # to_markdown de-wraps hard line breaks and infers headings from font
        # size; page.get_text() keeps visual wrapping, which strands sentences
        # mid-line and defeats sentence-boundary chunking.
        try:
            text = pymupdf4llm.to_markdown(doc, show_progress=False)
        except Exception:
            logger.warning("pdf_markdown_failed", extra={"source_filename": filename})
            text = "\n\n".join(page.get_text() for page in doc)

        char_count = len(text.strip())
        logger.info(
            "pdf_extracted",
            extra={"source_filename": filename, "pages": doc.page_count, "chars": char_count},
        )
        if char_count < _MIN_PDF_CHARS:
            raise ExtractionError("PDF appears to be scanned/image-based")
        return text, "pdf"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ExtractionError("File must be UTF-8 encoded text")

    if ext in (".html", ".htm"):
        return strip_html(text), "html"
    if ext == ".md":
        return text, "markdown"
    return text, "text"
