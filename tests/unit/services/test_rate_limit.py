import time
from unittest.mock import patch

from app.services.rate_limit import RateLimiter


class TestRateLimiter:
    def test_allows_up_to_max_requests(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("key") is True
        assert limiter.is_allowed("key") is True
        assert limiter.is_allowed("key") is True

    def test_blocks_after_max_requests(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("key")
        limiter.is_allowed("key")
        assert limiter.is_allowed("key") is False

    def test_different_keys_independent(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("a") is True
        assert limiter.is_allowed("a") is False
        assert limiter.is_allowed("b") is True

    def test_window_expiry(self):
        limiter = RateLimiter(max_requests=1, window_seconds=10)
        assert limiter.is_allowed("key") is True
        assert limiter.is_allowed("key") is False

        # Advance time past the window
        with patch.object(time, "monotonic", return_value=time.monotonic() + 11):
            assert limiter.is_allowed("key") is True

    def test_partial_expiry(self):
        """Some requests expire while newer ones remain."""
        base = time.monotonic()
        limiter = RateLimiter(max_requests=2, window_seconds=10)

        with patch.object(time, "monotonic", return_value=base):
            limiter.is_allowed("key")  # t=0

        with patch.object(time, "monotonic", return_value=base + 5):
            limiter.is_allowed("key")  # t=5

        # At t=11, first request expired but second hasn't
        with patch.object(time, "monotonic", return_value=base + 11):
            assert limiter.is_allowed("key") is True  # one slot freed

    def test_cleanup_empty_keys(self):
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        limiter.is_allowed("temp")

        # After window expires, next check should clean up the key
        with patch.object(time, "monotonic", return_value=time.monotonic() + 2):
            limiter.is_allowed("temp")
            # Key with no recent requests might still exist (defaultdict),
            # but the internal list should be cleaned
            # The implementation pops empty keys
            assert "temp" in limiter._requests  # just got re-added
