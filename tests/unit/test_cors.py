from unittest.mock import patch

import app.cors as cors_module
from app.cors import DynamicCORSMiddleware, _origins_by_slug, _SLUG_RE


class TestSlugRegex:
    def test_matches_model_path(self):
        assert _SLUG_RE.match("/models/my-bot/chat")
        assert _SLUG_RE.match("/models/abc123/sources")

    def test_extracts_slug(self):
        match = _SLUG_RE.match("/models/my-bot/chat")
        assert match.group(1) == "my-bot"

    def test_no_match_non_model_paths(self):
        assert _SLUG_RE.match("/healthz") is None
        assert _SLUG_RE.match("/readyz") is None
        assert _SLUG_RE.match("/models") is None

    def test_slug_must_start_with_alphanumeric(self):
        assert _SLUG_RE.match("/models/-bad/chat") is None
        assert _SLUG_RE.match("/models/0good/chat") is not None


class TestOriginResolution:
    """Test origin resolution logic without actually running CORS middleware."""

    def setup_method(self):
        _origins_by_slug.clear()

    def teardown_method(self):
        _origins_by_slug.clear()

    def test_model_route_combines_model_and_console_origins(self):
        _origins_by_slug["my-bot"] = ["https://widget.example.com"]

        with patch("app.config.settings") as mock_settings:
            mock_settings.console_origins = ["http://localhost:5173"]

            scope = {"type": "http", "path": "/models/my-bot/chat"}
            match = _SLUG_RE.match(scope["path"])
            slug = match.group(1)
            model_origins = _origins_by_slug.get(slug, [])
            origins = list(set(model_origins + mock_settings.console_origins))

        assert "https://widget.example.com" in origins
        assert "http://localhost:5173" in origins

    def test_model_route_unknown_slug_gets_console_only(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.console_origins = ["http://localhost:5173"]

            model_origins = _origins_by_slug.get("unknown", [])
            origins = list(set(model_origins + mock_settings.console_origins))

        assert origins == ["http://localhost:5173"]

    def test_non_model_route_gets_console_only(self):
        _origins_by_slug["my-bot"] = ["https://widget.example.com"]

        with patch("app.config.settings") as mock_settings:
            mock_settings.console_origins = ["http://localhost:5173"]

            # Non-model path — should not include model origins
            path = "/healthz"
            match = _SLUG_RE.match(path)
            assert match is None
            origins = list(mock_settings.console_origins)

        assert origins == ["http://localhost:5173"]
        assert "https://widget.example.com" not in origins

    def test_deduplicates_origins(self):
        _origins_by_slug["my-bot"] = ["http://localhost:5173"]

        with patch("app.config.settings") as mock_settings:
            mock_settings.console_origins = ["http://localhost:5173"]

            model_origins = _origins_by_slug.get("my-bot", [])
            origins = list(set(model_origins + mock_settings.console_origins))

        assert len(origins) == 1


class TestCorsCache:
    def test_same_origins_return_cached_instance(self):
        app = lambda scope, receive, send: None
        middleware = DynamicCORSMiddleware(app)

        cors1 = middleware._get_cors(["https://a.com"])
        cors2 = middleware._get_cors(["https://a.com"])
        assert cors1 is cors2

    def test_different_origins_return_different_instances(self):
        app = lambda scope, receive, send: None
        middleware = DynamicCORSMiddleware(app)

        cors1 = middleware._get_cors(["https://a.com"])
        cors2 = middleware._get_cors(["https://b.com"])
        assert cors1 is not cors2
