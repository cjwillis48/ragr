import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.crawler import _normalize_url, _extract_links, crawl_site


class TestNormalizeUrl:
    def test_strips_fragment(self):
        assert _normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_strips_trailing_slash(self):
        assert _normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_root_keeps_slash(self):
        assert _normalize_url("https://example.com/") == "https://example.com/"

    def test_no_fragment_no_trailing_slash(self):
        assert _normalize_url("https://example.com/page") == "https://example.com/page"

    def test_preserves_scheme_and_netloc(self):
        assert _normalize_url("http://sub.example.com:8080/path") == "http://sub.example.com:8080/path"


class TestExtractLinks:
    def test_same_domain_only(self):
        html = '<a href="https://example.com/page1">Link</a><a href="https://other.com/page">Other</a>'
        links = _extract_links(html, "https://example.com", "example.com", None)
        assert len(links) == 1
        assert "example.com/page1" in links[0]

    def test_relative_links_resolved(self):
        html = '<a href="/about">About</a>'
        links = _extract_links(html, "https://example.com/page", "example.com", None)
        assert links == ["https://example.com/about"]

    def test_non_http_filtered(self):
        html = '<a href="mailto:test@example.com">Email</a><a href="javascript:void(0)">JS</a>'
        links = _extract_links(html, "https://example.com", "example.com", None)
        assert links == []

    def test_prefix_filter(self):
        html = '<a href="/docs/api">API</a><a href="/blog/post">Blog</a>'
        links = _extract_links(html, "https://example.com", "example.com", "/docs")
        assert len(links) == 1
        assert "/docs/api" in links[0]

    def test_no_prefix_returns_all_same_domain(self):
        html = '<a href="/a">A</a><a href="/b">B</a>'
        links = _extract_links(html, "https://example.com", "example.com", None)
        assert len(links) == 2

    def test_fragments_stripped_in_output(self):
        html = '<a href="/page#section">Link</a>'
        links = _extract_links(html, "https://example.com", "example.com", None)
        assert "#" not in links[0]


class TestCrawlSite:
    def _mock_response(self, text: str, content_type: str = "text/html", status: int = 200, size: int = 100):
        resp = MagicMock()
        resp.text = text
        resp.content = b"x" * size
        resp.headers = {"content-type": content_type}
        resp.status_code = status
        resp.raise_for_status = MagicMock()
        if status >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
        return resp

    async def test_basic_crawl(self):
        html = '<html><body><p>Hello world content that is long enough to pass the 50 char minimum threshold.</p></body></html>'
        mock_get = AsyncMock(return_value=self._mock_response(html))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.safe_get", mock_get)
            results = await crawl_site("https://example.com", max_pages=5)

        assert len(results) == 1
        assert results[0].url == "https://example.com/"
        assert "Hello world" in results[0].text

    async def test_max_pages_limit(self):
        html = '<html><body><p>Content that is long enough to pass the 50 char minimum threshold for a crawled page.</p><a href="/page2">Next</a></body></html>'
        mock_get = AsyncMock(return_value=self._mock_response(html))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.safe_get", mock_get)
            results = await crawl_site("https://example.com", max_pages=1)

        assert len(results) == 1

    async def test_skips_non_html(self):
        resp = self._mock_response("binary data", content_type="application/pdf")
        mock_get = AsyncMock(return_value=resp)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.safe_get", mock_get)
            results = await crawl_site("https://example.com", max_pages=5)

        assert len(results) == 0

    async def test_skips_short_text(self):
        html = "<html><body><p>Short</p></body></html>"  # < 50 chars of text
        mock_get = AsyncMock(return_value=self._mock_response(html))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.safe_get", mock_get)
            results = await crawl_site("https://example.com")

        assert len(results) == 0

    async def test_skips_oversized_page(self):
        html = '<html><body><p>Content</p></body></html>'
        resp = self._mock_response(html, size=11 * 1024 * 1024)  # > 10MB
        mock_get = AsyncMock(return_value=resp)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.safe_get", mock_get)
            results = await crawl_site("https://example.com")

        assert len(results) == 0

    async def test_exclude_patterns(self):
        html = '<html><body><p>Content that is long enough to pass the 50 char minimum threshold for crawling.</p></body></html>'
        mock_get = AsyncMock(return_value=self._mock_response(html))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.safe_get", mock_get)
            results = await crawl_site(
                "https://example.com/admin/page",
                exclude_patterns=["*/admin/*"],
            )

        assert len(results) == 0

    async def test_handles_fetch_errors(self):
        mock_get = AsyncMock(side_effect=Exception("connection failed"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.crawler.safe_get", mock_get)
            results = await crawl_site("https://example.com")

        assert len(results) == 0
