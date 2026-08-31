"""Same-domain URL crawler for site-wide ingestion."""

import asyncio
import fnmatch
import logging
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from app.services.html import parse_html
from app.services.url_validation import safe_get, validate_url
from app.services.wikipedia import fetch_wikipedia_html, is_wikipedia_domain, is_wikipedia_url, parse_wikipedia_url

# Common pattern: redirects to add/remove a trailing slash, http→https, www→apex.
# 5 hops covers nearly all real-world chains.
_MAX_REDIRECTS = 5

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


@dataclass
class SkippedPage:
    """A page fetched successfully but not ingested, and why.

    Distinct from FailedPage: nothing went wrong on the wire. The most common
    reason is `empty_render` — the fetch returned HTML with no text and no
    links, which is what a client-side rendered app looks like to a crawler
    that does not run JavaScript.
    """
    url: str
    reason: str  # not_html | oversized | empty_render | thin
    detail: str | None = None


def normalize_url(url: str) -> str:
    """Canonical form for dedup: strip fragment, ensure exactly one path slash.

    Both `https://x.com` and `https://x.com/` collapse to `https://x.com/`,
    so the API layer and the crawler agree on the same source_identifier.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"




async def _fetch_page(url: str, timeout: float = 30):
    """Fetch a page, using Wikipedia API for Wikipedia URLs.

    Manually follows up to _MAX_REDIRECTS hops because safehttpx pins
    follow_redirects=False (each hop could land on a private IP). Each
    redirect target is re-validated through validate_url() before fetch.
    """
    wp = parse_wikipedia_url(url)
    if wp:
        lang, title = wp
        return await fetch_wikipedia_html(lang, title, timeout=timeout)

    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        resp = await safe_get(current, timeout=timeout)
        if resp.status_code not in (301, 302, 303, 307, 308):
            return resp
        location = resp.headers.get("location")
        if not location:
            return resp  # let the caller's raise_for_status handle this odd case
        current = urljoin(current, location)
        await validate_url(current)
    raise RuntimeError(f"Exceeded {_MAX_REDIRECTS} redirects starting from {url}")


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
    skipped_count = 0

    start = normalize_url(root_url)
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

        # Fetch, parse, and extract links concurrently per batch
        import time as _time

        def _process_html(raw_html: str, url: str):
            """CPU-bound: parse HTML once for text and links."""
            return parse_html(raw_html, url, domain, prefix)

        async def _process_url(url: str, depth: int):
            t_fetch = _time.perf_counter()
            try:
                resp = await _fetch_page(url, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                logger.warning("crawl_fetch_failed", extra={"url": url, "error": str(e), "fetch_ms": round((_time.perf_counter() - t_fetch) * 1000)}, exc_info=True)
                return FailedPage(url=url, error=str(e))
            fetch_ms = round((_time.perf_counter() - t_fetch) * 1000)

            if len(resp.content) > 10 * 1024 * 1024:
                logger.warning("crawl_page_oversized", extra={"url": url, "bytes": len(resp.content)})
                return SkippedPage(url=url, reason="oversized", detail=f"{len(resp.content) // (1024 * 1024)} MB")

            content_type_header = resp.headers.get("content-type", "")
            if "html" not in content_type_header:
                return SkippedPage(url=url, reason="not_html", detail=content_type_header or "unknown")

            t_parse = _time.perf_counter()
            raw_html = resp.text
            text, links = await asyncio.to_thread(_process_html, raw_html, url)
            parse_ms = round((_time.perf_counter() - t_parse) * 1000)

            if not text or len(text) < 50:
                # No text AND no links means nothing was server-rendered at all —
                # the signature of a client-side app, not of a thin page.
                reason = "empty_render" if not links else "thin"
                return SkippedPage(url=url, reason=reason, detail=f"{len(text)} chars")

            return (CrawledPage(url=url, text=text, content_type="html"), links, depth, fetch_ms, parse_ms)

        t_batch = _time.perf_counter()
        results = await asyncio.gather(*[_process_url(url, depth) for url, depth in batch])
        batch_ms = round((_time.perf_counter() - t_batch) * 1000)
        logger.info("crawl_batch_done", extra={"batch_size": len(batch), "batch_ms": batch_ms})

        for result in results:
            if result is None:
                continue
            if isinstance(result, FailedPage):
                failed_count += 1
                yield result
                continue
            if isinstance(result, SkippedPage):
                skipped_count += 1
                logger.info("crawl_page_skipped", extra={"url": result.url, "reason": result.reason, "detail": result.detail})
                yield result
                continue

            page, links, depth, fetch_ms, parse_ms = result
            page_count += 1

            # Add discovered links to queue
            t_links = _time.perf_counter()
            new_links = 0
            if depth < max_depth:
                for link in links:
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
                        new_links += 1
            links_ms = round((_time.perf_counter() - t_links) * 1000)

            logger.info("crawled_page", extra={
                "url": page.url, "chars": len(page.text), "depth": depth,
                "page": page_count, "max_pages": max_pages,
                "fetch_ms": fetch_ms, "parse_ms": parse_ms, "links_ms": links_ms, "new_links": new_links,
            })
            yield page

    logger.info("crawl_complete", extra={"pages": page_count, "failed": failed_count, "skipped": skipped_count, "root_url": root_url})


def explain_empty_crawl(reason: str | None, detail: str | None = None) -> str:
    """Human-readable reason a crawl ingested nothing, for the console.

    Lives next to the reason strings so the copy can't drift from the codes.
    """
    if reason == "empty_render":
        return (
            "Fetched the page successfully, but it contained no readable text and no links. "
            "This site renders its content in the browser (a JavaScript app), and RAGr reads "
            "only the HTML the server sends. Add your content directly instead."
        )
    if reason == "thin":
        return f"Fetched the page, but it had too little readable text to index ({detail})."
    if reason == "not_html":
        return f"The URL returned {detail}, not an HTML page."
    if reason == "oversized":
        return f"The page was larger than the 10 MB limit ({detail})."
    if reason == "fetch_failed":
        return f"Could not fetch the page: {detail}"
    return "The crawl finished without finding any readable pages."
