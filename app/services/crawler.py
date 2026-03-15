"""Same-domain URL crawler for site-wide ingestion."""

import logging
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("ragr.crawler")


@dataclass
class CrawledPage:
    url: str
    text: str
    content_type: str


def _normalize_url(url: str) -> str:
    """Strip fragment and trailing slash for dedup."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _extract_links(html: str, base_url: str, domain: str, prefix: str | None) -> list[str]:
    """Extract same-domain links from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        # Same domain only
        if parsed.netloc != domain:
            continue

        # Skip non-HTTP
        if parsed.scheme not in ("http", "https"):
            continue

        # Optional prefix filter
        if prefix and not parsed.path.startswith(prefix):
            continue

        normalized = _normalize_url(absolute)
        links.append(normalized)

    return links


async def crawl_site(
    root_url: str,
    max_pages: int = 50,
    max_depth: int = 3,
    prefix: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[CrawledPage]:
    """Crawl a site starting from root_url, returning discovered pages.

    Returns list of CrawledPage results.
    """
    parsed_root = urlparse(root_url)
    domain = parsed_root.netloc
    if prefix is None and parsed_root.path and parsed_root.path != "/":
        # Auto-scope to the root URL's path prefix
        prefix = None  # Don't auto-restrict, let user opt in

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    results: list[CrawledPage] = []

    start = _normalize_url(root_url)
    queue.append((start, 0))
    visited.add(start)

    excludes = exclude_patterns or []

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        while queue and len(results) < max_pages:
            url, depth = queue.popleft()

            # Check exclude patterns
            if any(pattern in url for pattern in excludes):
                continue

            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception:
                logger.warning("Failed to fetch %s, skipping", url)
                continue

            content_type_header = resp.headers.get("content-type", "")
            if "html" not in content_type_header:
                continue

            raw_html = resp.text
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "head"]):
                tag.decompose()
            text = soup.get_text(separator="\n\n").strip()

            if not text or len(text) < 50:
                continue

            results.append(CrawledPage(url=url, text=text, content_type="html"))
            logger.info("Crawled %s (%d chars, depth %d, %d/%d pages)", url, len(text), depth, len(results), max_pages)

            # Discover links if we haven't hit depth limit
            if depth < max_depth:
                for link in _extract_links(raw_html, url, domain, prefix):
                    if link not in visited:
                        visited.add(link)
                        queue.append((link, depth + 1))

    logger.info("Crawl complete: %d pages from %s", len(results), root_url)
    return results
