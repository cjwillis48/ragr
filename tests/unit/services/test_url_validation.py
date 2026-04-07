import socket
from unittest.mock import patch, AsyncMock

import pytest

from app.services.url_validation import validate_url, SSRFError


def _fake_getaddrinfo(ip: str):
    """Return a mock getaddrinfo result for the given IP."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


class TestValidateUrl:
    async def test_valid_public_url(self):
        with patch("app.services.url_validation.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = _fake_getaddrinfo("93.184.216.34")
            await validate_url("https://example.com/page")

    async def test_missing_scheme(self):
        with pytest.raises(SSRFError, match="missing scheme or host"):
            await validate_url("example.com")

    async def test_non_http_scheme(self):
        with pytest.raises(SSRFError, match="Only http and https"):
            await validate_url("ftp://example.com")

    async def test_missing_hostname(self):
        with pytest.raises(SSRFError, match="missing scheme or host"):
            await validate_url("https://")

    async def test_private_ip_blocked(self):
        for ip in ["10.0.0.1", "192.168.1.1", "172.16.0.1"]:
            with patch("app.services.url_validation.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = _fake_getaddrinfo(ip)
                with pytest.raises(SSRFError, match="private/reserved"):
                    await validate_url("https://evil.com")

    async def test_loopback_blocked(self):
        with patch("app.services.url_validation.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = _fake_getaddrinfo("127.0.0.1")
            with pytest.raises(SSRFError, match="private/reserved"):
                await validate_url("https://localhost")

    async def test_link_local_blocked(self):
        with patch("app.services.url_validation.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = _fake_getaddrinfo("169.254.1.1")
            with pytest.raises(SSRFError, match="private/reserved"):
                await validate_url("https://evil.com")

    async def test_dns_failure(self):
        with patch("app.services.url_validation.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = socket.gaierror("DNS failed")
            with pytest.raises(SSRFError, match="Cannot resolve hostname"):
                await validate_url("https://nonexistent.example")

    async def test_no_addresses(self):
        with patch("app.services.url_validation.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = []
            with pytest.raises(SSRFError, match="No addresses found"):
                await validate_url("https://example.com")

    async def test_http_scheme_allowed(self):
        with patch("app.services.url_validation.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = _fake_getaddrinfo("93.184.216.34")
            await validate_url("http://example.com")
