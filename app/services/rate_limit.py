"""Simple in-memory rate limiter for chat endpoints.

NOTE: This only works with a single replica. If we scale to multiple
replicas, replace with a shared store (e.g. Redis or similar).
"""

import time
from collections import defaultdict


class RateLimiter:
    """Token-bucket rate limiter keyed by an arbitrary string (IP, session, etc.)."""

    def __init__(self, max_requests: int, window_seconds: int = 60, max_keys: int = 10_000):
        self._max = max_requests
        self._window = window_seconds
        self._max_keys = max_keys
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if a request from `key` is allowed. Records the request if allowed."""
        now = time.monotonic()
        cutoff = now - self._window

        # Prune expired entries, then clean up keys with no recent requests
        recent = [t for t in self._requests[key] if t > cutoff]
        if not recent:
            self._requests.pop(key, None)

        # Evict stale keys if the dict has grown too large
        if len(self._requests) > self._max_keys:
            stale = [k for k, ts in self._requests.items() if not ts or ts[-1] <= cutoff]
            for k in stale:
                del self._requests[k]

        if len(recent) >= self._max:
            return False

        recent.append(now)
        self._requests[key] = recent
        return True
