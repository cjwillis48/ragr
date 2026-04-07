import time

from app.services.client_cache import ClientCache


class TestClientCache:
    def test_platform_client_created_once(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"platform-{call_count}"

        cache = ClientCache(platform_factory=factory, custom_factory=lambda k: k)
        assert cache.get() == "platform-1"
        assert cache.get() == "platform-1"
        assert call_count == 1

    def test_custom_client_cached_by_key(self):
        call_count = 0

        def factory(key):
            nonlocal call_count
            call_count += 1
            return f"client-{key}-{call_count}"

        cache = ClientCache(platform_factory=lambda: "platform", custom_factory=factory)
        first = cache.get("key-a")
        second = cache.get("key-a")
        assert first == second
        assert call_count == 1

    def test_different_keys_get_different_clients(self):
        cache = ClientCache(
            platform_factory=lambda: "platform",
            custom_factory=lambda k: f"client-{k}",
        )
        assert cache.get("key-a") != cache.get("key-b")

    def test_ttl_expiry(self, monkeypatch):
        fake_time = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

        call_count = 0

        def factory(key):
            nonlocal call_count
            call_count += 1
            return f"client-{call_count}"

        cache = ClientCache(platform_factory=lambda: "p", custom_factory=factory, ttl=10)

        first = cache.get("key")
        assert first == "client-1"

        # Still within TTL
        fake_time[0] = 9.0
        assert cache.get("key") == "client-1"

        # Past TTL
        fake_time[0] = 11.0
        second = cache.get("key")
        assert second == "client-2"
        assert call_count == 2

    def test_max_size_eviction(self):
        cache = ClientCache(
            platform_factory=lambda: "platform",
            custom_factory=lambda k: f"client-{k}",
            max_size=2,
        )
        cache.get("key-a")
        cache.get("key-b")
        # This should evict the oldest (key-a)
        cache.get("key-c")
        assert len(cache._cache) == 2
        # key-a was evicted, so accessing it creates a new client
        assert cache.get("key-a") == "client-key-a"
        assert len(cache._cache) == 2
