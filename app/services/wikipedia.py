"""Wikipedia API integration for fetching article content.

Uses the Wikipedia REST API instead of direct page scraping,
which avoids SSRF-protection issues with safehttpx (Wikipedia
rejects direct-IP connections due to virtual hosting).
"""

import re
from urllib.parse import unquote, urlparse

import httpx

_WIKIPEDIA_HOST_RE = re.compile(r"^([a-z]{2,3}(?:-[a-z]+)*)\.wikipedia\.org$")
_WIKI_PATH_RE = re.compile(r"^/wiki/(.+)$")

_USER_AGENT = "ragr/1.0 (RAG ingestion bot; https://github.com/cjwillis48/ragr)"


_SKIP_NAMESPACES = {"File:", "Category:", "Template:", "Wikipedia:", "Help:", "Talk:", "Special:", "Portal:", "Draft:"}


def parse_wikipedia_url(url: str) -> tuple[str, str] | None:
    """Extract (lang, title) from a Wikipedia article URL, or None if not a Wikipedia article.

    Skips non-article namespaces (File:, Category:, Template:, etc.).
    """
    parsed = urlparse(url)
    host_match = _WIKIPEDIA_HOST_RE.match(parsed.netloc)
    if not host_match:
        return None
    path_match = _WIKI_PATH_RE.match(parsed.path)
    if not path_match:
        return None
    lang = host_match.group(1)
    title = unquote(path_match.group(1))
    if any(title.startswith(ns) for ns in _SKIP_NAMESPACES):
        return None
    return (lang, title)


def is_wikipedia_url(url: str) -> bool:
    """True if URL is a crawlable Wikipedia article (not File:, Category:, etc.)."""
    return parse_wikipedia_url(url) is not None


def is_wikipedia_domain(url: str) -> bool:
    """True if URL is on *.wikipedia.org, regardless of namespace."""
    parsed = urlparse(url)
    return bool(_WIKIPEDIA_HOST_RE.match(parsed.netloc))


_wikipedia_client: httpx.AsyncClient | None = None


def _get_wikipedia_client() -> httpx.AsyncClient:
    """Reuse a single HTTP client for Wikipedia requests (connection pooling + TLS reuse)."""
    global _wikipedia_client
    if _wikipedia_client is None or _wikipedia_client.is_closed:
        _wikipedia_client = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=30,
        )
    return _wikipedia_client


async def fetch_wikipedia_html(lang: str, title: str, timeout: float = 30) -> httpx.Response:
    """Fetch article HTML from the Wikipedia REST API."""
    api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
    client = _get_wikipedia_client()
    return await client.get(api_url, timeout=timeout)
