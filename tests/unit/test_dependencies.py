from app.dependencies import _extract_bearer, ClerkUser


class TestExtractBearer:
    def test_valid_bearer(self):
        assert _extract_bearer("Bearer abc123") == "abc123"

    def test_none(self):
        assert _extract_bearer(None) is None

    def test_no_prefix(self):
        assert _extract_bearer("abc123") is None

    def test_empty_string(self):
        assert _extract_bearer("") is None

    def test_bearer_only(self):
        assert _extract_bearer("Bearer ") == ""

    def test_case_sensitive(self):
        assert _extract_bearer("bearer abc123") is None


class TestClerkUser:
    def test_is_superuser_true(self):
        user = ClerkUser(user_id="superuser_123")
        assert user.is_superuser is True

    def test_is_superuser_false(self):
        user = ClerkUser(user_id="regular_user")
        assert user.is_superuser is False

    def test_email_optional(self):
        user = ClerkUser(user_id="user_1")
        assert user.email is None

    def test_email_set(self):
        user = ClerkUser(user_id="user_1", email="test@example.com")
        assert user.email == "test@example.com"
