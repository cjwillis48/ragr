"""HTML text extraction utilities shared by URL ingestion and crawling."""

from bs4 import BeautifulSoup

_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "head"]


def strip_html(raw_html: str) -> str:
    """Remove boilerplate tags and return clean text from HTML."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    return soup.get_text(separator="\n\n").strip()
