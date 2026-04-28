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


def parse_wikipedia_url(url: str) -> tuple[str, str] | None:
    """Extract (lang, title) from a Wikipedia URL, or None if not Wikipedia."""
    parsed = urlparse(url)
    host_match = _WIKIPEDIA_HOST_RE.match(parsed.netloc)
    if not host_match:
        return None
    path_match = _WIKI_PATH_RE.match(parsed.path)
    if not path_match:
        return None
    lang = host_match.group(1)
    title = unquote(path_match.group(1))
    return (lang, title)


def is_wikipedia_url(url: str) -> bool:
    return parse_wikipedia_url(url) is not None


async def fetch_wikipedia_html(lang: str, title: str, timeout: float = 30) -> httpx.Response:
    """Fetch article HTML from the Wikipedia REST API."""
    api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
    async with httpx.AsyncClient() as client:
        return await client.get(
            api_url,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
