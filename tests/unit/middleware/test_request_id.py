import uuid

from app.middleware.request_id import _SAFE_ID_RE, REQUEST_ID_CTX


class TestSafeIdRegex:
    def test_valid_uuid(self):
        assert _SAFE_ID_RE.match(str(uuid.uuid4()))

    def test_valid_alphanumeric(self):
        assert _SAFE_ID_RE.match("abc-123_def.456")

    def test_rejects_newlines(self):
        assert _SAFE_ID_RE.match("abc\ndef") is None

    def test_rejects_spaces(self):
        assert _SAFE_ID_RE.match("abc def") is None

    def test_rejects_html(self):
        assert _SAFE_ID_RE.match("<script>alert(1)</script>") is None

    def test_rejects_empty(self):
        assert _SAFE_ID_RE.match("") is None

    def test_rejects_too_long(self):
        assert _SAFE_ID_RE.match("a" * 129) is None

    def test_accepts_max_length(self):
        assert _SAFE_ID_RE.match("a" * 128)

    def test_rejects_log_injection(self):
        # Newline-based log injection attempt
        assert _SAFE_ID_RE.match('abc\n2026-04-01 CRITICAL fake log') is None


class TestRequestIdContext:
    def test_default_value(self):
        assert REQUEST_ID_CTX.get("-") == "-"

    def test_set_and_get(self):
        token = REQUEST_ID_CTX.set("test-id-123")
        assert REQUEST_ID_CTX.get() == "test-id-123"
        REQUEST_ID_CTX.reset(token)
