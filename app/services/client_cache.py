"""TTL-bounded cache for API clients keyed by custom API key.

Used by embedder, reranker, and generation services to reuse clients
for custom API keys without unbounded memory growth.
"""

import hashlib
import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

_DEFAULT_TTL = 300  # 5 minutes
_DEFAULT_MAX_SIZE = 64


class ClientCache(Generic[T]):
    """Cache that maps custom API keys to client instances with TTL and size cap.

    Usage:
        cache = ClientCache(
            platform_factory=lambda: AsyncAnthropic(api_key=settings.key),
            custom_factory=lambda key: AsyncAnthropic(api_key=key),
        )
        client = cache.get()           # platform client
        client = cache.get(custom_key) # cached custom-key client
    """

    def __init__(
        self,
        platform_factory: Callable[[], T],
        custom_factory: Callable[[str], T],
        ttl: int = _DEFAULT_TTL,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self._platform_factory = platform_factory
        self._custom_factory = custom_factory
        self._ttl = ttl
        self._max_size = max_size
        self._platform_client: T | None = None
        self._cache: dict[str, tuple[T, float]] = {}

    def get(self, api_key: str | None = None) -> T:
        if api_key:
            return self._get_custom(api_key)
        if self._platform_client is None:
            self._platform_client = self._platform_factory()
        return self._platform_client

    def _get_custom(self, api_key: str) -> T:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = time.monotonic()
        entry = self._cache.get(key_hash)
        if entry and (now - entry[1]) < self._ttl:
            return entry[0]

        # Evict oldest entry if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        client = self._custom_factory(api_key)
        self._cache[key_hash] = (client, now)
        return client
