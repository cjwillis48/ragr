"""HTML text extraction utilities shared by URL ingestion and crawling."""

from urllib.parse import urljoin, urlparse

from selectolax.lexbor import LexborHTMLParser

_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "head"]


def _strip_boilerplate(tree: LexborHTMLParser) -> None:
    for tag in _BOILERPLATE_TAGS:
        for node in tree.css(tag):
            node.decompose()


def strip_html(raw_html: str) -> str:
    """Remove boilerplate tags and return clean text from HTML."""
    tree = LexborHTMLParser(raw_html)
    _strip_boilerplate(tree)
    root = tree.body or tree.root
    if root is None:
        return ""
    return root.text(separator="\n\n").strip()


def parse_html(raw_html: str, base_url: str, domain: str, prefix: str | None) -> tuple[str, list[str]]:
    """Parse HTML once, returning both clean text and same-domain links."""
    tree = LexborHTMLParser(raw_html)

    # Extract links before stripping boilerplate (nav links are still useful)
    links = []
    for a in tree.css("a[href]"):
        href = a.attributes.get("href")
        if not href:
            continue
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

    _strip_boilerplate(tree)
    root = tree.body or tree.root
    text = root.text(separator="\n\n").strip() if root is not None else ""

    return text, links