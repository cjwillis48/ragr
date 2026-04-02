"""Top-level conftest — sets env vars BEFORE any app imports.

Uses setdefault so that env vars passed externally (e.g. DATABASE_URL
for integration tests) take precedence over these defaults.
"""

import os

from cryptography.fernet import Fernet

_DEFAULTS = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/test",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "VOYAGE_API_KEY": "test-voyage-key",
    "ENCRYPTION_KEY": Fernet.generate_key().decode(),
    "CLERK_SECRET_KEY": "test-clerk-key",
    "SUPERUSER_ID": "superuser_123",
    "R2_ACCOUNT_ID": "",
    "R2_ACCESS_KEY_ID": "",
    "R2_SECRET_ACCESS_KEY": "",
    "RATE_LIMIT_PER_MIN": "100",
    "CONSOLE_ORIGINS": '[]',
}

for key, value in _DEFAULTS.items():
    os.environ.setdefault(key, value)
