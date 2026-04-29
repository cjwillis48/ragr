"""HTML text extraction utilities shared by URL ingestion and crawling."""

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "head"]


def strip_html(raw_html: str) -> str:
    """Remove boilerplate tags and return clean text from HTML."""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    return soup.get_text(separator="\n\n").strip()


def parse_html(raw_html: str, base_url: str, domain: str, prefix: str | None) -> tuple[str, list[str]]:
    """Parse HTML once, returning both clean text and same-domain links."""
    soup = BeautifulSoup(raw_html, "lxml")

    # Extract links before stripping boilerplate (nav links are still useful)
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        if parsed.netloc != domain:
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        if prefix and not parsed.path.startswith(prefix):
            continue

        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/') or '/'}"
        links.append(normalized)

    # Strip boilerplate for text extraction
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n\n").strip()

    return text, links
