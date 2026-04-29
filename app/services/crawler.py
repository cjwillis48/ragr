"""Same-domain URL crawler for site-wide ingestion."""

import asyncio
import fnmatch
import logging
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.services.html import strip_html
from app.services.url_validation import safe_get, validate_url
from app.services.wikipedia import fetch_wikipedia_html, is_wikipedia_domain, is_wikipedia_url, parse_wikipedia_url

logger = logging.getLogger("ragr.crawler")


@dataclass
class CrawledPage:
    url: str
    text: str
    content_type: str


@dataclass
class FailedPage:
    url: str
    error: str


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


async def _fetch_page(url: str, timeout: float = 30):
    """Fetch a page, using Wikipedia API for Wikipedia URLs."""
    wp = parse_wikipedia_url(url)
    if wp:
        lang, title = wp
        return await fetch_wikipedia_html(lang, title, timeout=timeout)
    return await safe_get(url, timeout=timeout)


async def crawl_site(
    root_url: str,
    max_pages: int = 50,
    max_depth: int = 3,
    prefix: str | None = None,
    exclude_patterns: list[str] | None = None,
):
    """Crawl a site starting from root_url.

    Async generator — yields CrawledPage or FailedPage as each URL is processed.
    Only one page of text is held in memory at a time.
    """
    parsed_root = urlparse(root_url)
    domain = parsed_root.netloc

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    page_count = 0
    failed_count = 0

    start = _normalize_url(root_url)
    queue.append((start, 0))
    visited.add(start)

    excludes = exclude_patterns or []

    fetch_batch_size = 5

    while queue and page_count < max_pages:
        # Pop a batch of URLs from the queue
        batch = []
        while queue and len(batch) < fetch_batch_size and page_count + len(batch) < max_pages:
            url, depth = queue.popleft()
            if any(fnmatch.fnmatch(url, pattern) for pattern in excludes):
                continue
            batch.append((url, depth))

        if not batch:
            break

        # Fetch all pages in the batch concurrently
        async def _process_url(url: str, depth: int):
            try:
                resp = await _fetch_page(url, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                logger.warning("crawl_fetch_failed", extra={"url": url, "error": str(e)}, exc_info=True)
                return FailedPage(url=url, error=str(e))

            if len(resp.content) > 10 * 1024 * 1024:
                logger.warning("crawl_page_oversized", extra={"url": url, "bytes": len(resp.content)})
                return None

            content_type_header = resp.headers.get("content-type", "")
            if "html" not in content_type_header:
                return None

            raw_html = resp.text
            text = await asyncio.to_thread(strip_html, raw_html)

            if not text or len(text) < 50:
                return None

            # Return page + raw HTML for link extraction
            return (CrawledPage(url=url, text=text, content_type="html"), raw_html, depth)

        results = await asyncio.gather(*[_process_url(url, depth) for url, depth in batch])

        for result in results:
            if result is None:
                continue
            if isinstance(result, FailedPage):
                failed_count += 1
                yield result
                continue

            page, raw_html, depth = result
            page_count += 1
            logger.info("crawled_page", extra={"url": page.url, "chars": len(page.text), "depth": depth, "page": page_count, "max_pages": max_pages})
            yield page

            # Discover links if we haven't hit depth limit
            if depth < max_depth:
                for link in await asyncio.to_thread(_extract_links, raw_html, page.url, domain, prefix):
                    if link not in visited:
                        if is_wikipedia_domain(link) and not is_wikipedia_url(link):
                            continue
                        if not is_wikipedia_url(link):
                            try:
                                await validate_url(link)
                            except ValueError:
                                logger.debug("crawl_link_rejected", extra={"url": link})
                                continue
                        visited.add(link)
                        queue.append((link, depth + 1))

    logger.info("crawl_complete", extra={"pages": page_count, "failed": failed_count, "root_url": root_url})
