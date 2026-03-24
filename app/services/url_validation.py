"""URL validation utilities to prevent SSRF attacks."""

import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_SCHEMES = frozenset({"file", "ftp", "data", "javascript", "gopher"})


class SSRFError(ValueError):
    """Raised when a URL targets a private or blocked address."""


def validate_url(url: str) -> str:
    """Validate a URL is safe to fetch.

    Checks:
    - Scheme is http or https
    - Hostname resolves to a public (non-private, non-loopback, non-link-local) IP

    Returns the validated URL. Raises SSRFError on failure.
    """
    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise SSRFError(f"Invalid URL: {url}")

    if parsed.scheme not in ("http", "https"):
        raise SSRFError(f"Blocked URL scheme '{parsed.scheme}': only http/https allowed")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError(f"No hostname in URL: {url}")

    # Resolve hostname to IP addresses and check each one
    try:
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        raise SSRFError(f"Cannot resolve hostname: {hostname}")

    if not addr_infos:
        raise SSRFError(f"No addresses found for hostname: {hostname}")

    for family, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise SSRFError(
                f"Blocked request to private/reserved address: {hostname} -> {ip}"
            )

    return url
