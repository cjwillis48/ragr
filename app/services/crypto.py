"""Symmetric encryption for sensitive fields (API keys) using Fernet."""

from cryptography.fernet import Fernet

from app.config import settings

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        if not settings.encryption_key:
            raise RuntimeError("ENCRYPTION_KEY is not configured")
        _fernet = Fernet(settings.encryption_key.encode())
    return _fernet


def encrypt(value: str) -> str:
    """Encrypt a plaintext string. Returns a base64-encoded ciphertext."""
    return _get_fernet().encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    """Decrypt a Fernet-encrypted string."""
    return _get_fernet().decrypt(value.encode()).decode()
