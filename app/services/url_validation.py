"""URL validation utilities to prevent SSRF attacks."""

import ipaddress
import socket
from urllib.parse import urlparse


class SSRFError(ValueError):
    """Raised when a URL targets a private or blocked address."""


def _check_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, hostname: str) -> None:
    """Raise SSRFError if the IP is private/reserved."""
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise SSRFError(
            f"Blocked request to private/reserved address: {hostname} -> {ip}"
        )


def validate_url(url: str) -> tuple[str, list[str]]:
    """Validate a URL is safe to fetch.

    Checks:
    - Scheme is http or https
    - Hostname resolves to a public (non-private, non-loopback, non-link-local) IP

    Returns (validated_url, resolved_ips). Raises SSRFError on failure.
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

    resolved_ips: list[str] = []
    for family, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        _check_ip(ip, hostname)
        resolved_ips.append(str(ip))

    return url, resolved_ips


def check_response_ip(peer_ip: str, hostname: str = "unknown") -> None:
    """Re-validate the actual IP after an HTTP connection to defeat DNS rebinding.

    Call this with the peer address from the HTTP response/connection.
    """
    ip = ipaddress.ip_address(peer_ip)
    _check_ip(ip, hostname)
