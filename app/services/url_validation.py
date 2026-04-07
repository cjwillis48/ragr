"""URL validation and SSRF-safe HTTP fetching.

Uses safehttpx for DNS-pinned connections that prevent DNS rebinding.
validate_url() is kept for fast upfront input validation at the API layer
(before spawning background tasks).
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import safehttpx


class SSRFError(ValueError):
    """Raised when a URL targets a private or blocked address."""


async def validate_url(url: str) -> None:
    """Validate a URL is safe to fetch (upfront, before background work).

    Checks scheme, hostname presence, DNS resolution, and IP safety.
    Raises SSRFError on failure.
    """
    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise SSRFError("Invalid URL: missing scheme or host")

    if parsed.scheme not in ("http", "https"):
        raise SSRFError("Only http and https URLs are allowed")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("Invalid URL: no hostname")

    try:
        addr_infos = await asyncio.to_thread(
            socket.getaddrinfo, hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror:
        raise SSRFError("Cannot resolve hostname")

    if not addr_infos:
        raise SSRFError("No addresses found for hostname")

    for family, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise SSRFError("Blocked request to private/reserved address")


async def safe_get(url: str, **kwargs) -> "safehttpx.httpx.Response":
    """SSRF-safe HTTP GET. Thin wrapper around safehttpx.get()."""
    return await safehttpx.get(url, **kwargs)
