"""Simple in-memory rate limiter for chat endpoints."""

import time
from collections import defaultdict


class RateLimiter:
    """Token-bucket rate limiter keyed by an arbitrary string (IP, session, etc.)."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if a request from `key` is allowed. Records the request if allowed."""
        now = time.monotonic()
        cutoff = now - self._window

        # Prune expired entries
        timestamps = self._requests[key]
        self._requests[key] = [t for t in timestamps if t > cutoff]

        if len(self._requests[key]) >= self._max:
            return False

        self._requests[key].append(now)
        return True
