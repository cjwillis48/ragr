"""HTML text extraction utilities shared by URL ingestion and crawling."""

import re
from urllib.parse import urljoin, urlparse

from markdownify import markdownify
from selectolax.lexbor import LexborHTMLParser

_BOILERPLATE_TAGS = [
    "script", "style", "nav", "footer", "head", "aside", "form", "header", "noscript",
]

# Wrappers that carry navigation, infoboxes and citation scaffolding rather than
# prose. Left in, they chunk into hundreds of one-word fragments.
_BOILERPLATE_SELECTORS = [
    ".navbox", ".infobox", ".sidebar", ".metadata", ".mw-editsection",
    ".reference", ".mw-references-wrap", ".reflist", ".toc", ".hatnote",
    "[role=navigation]", "[role=complementary]", "[role=banner]",
]

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def _strip_boilerplate(tree: LexborHTMLParser) -> None:
    for selector in _BOILERPLATE_TAGS + _BOILERPLATE_SELECTORS:
        for node in tree.css(selector):
            node.decompose()


def _to_markdown(tree: LexborHTMLParser) -> str:
    """Serialize a cleaned tree to Markdown, preserving block structure.

    Block-level rather than text-node-level: headings become `#`, list items
    become `-`, and inline elements stay inside their sentence. Anchors and
    images are unwrapped to their text so URLs don't bloat the chunks — the
    crawler harvests links separately in parse_html.
    """
    root = tree.body or tree.root
    if root is None:
        return ""
    text = markdownify(root.html or "", heading_style="ATX", strip=["a", "img"])
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


def strip_html(raw_html: str) -> str:
    """Remove boilerplate and return clean Markdown text from HTML."""
    tree = LexborHTMLParser(raw_html)
    _strip_boilerplate(tree)
    return _to_markdown(tree)


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
    return _to_markdown(tree), links
