"""Logging context: ContextVars and filter for structured log enrichment.

REQUEST_ID_CTX is set by RequestIdMiddleware.
The tenant lives in app.tenancy (it drives row-level security, not just logs)
and is read here so every record carries the model it belongs to.

The LogContextFilter injects both into every log record automatically.
"""

import logging
from contextvars import ContextVar

from app.tenancy import current_model_id

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")


class LogContextFilter(logging.Filter):
    """Injects request_id and model_id from context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get("-")  # type: ignore[attr-defined]
        model_id = current_model_id()
        if model_id is not None:
            record.model_id = model_id  # type: ignore[attr-defined]
        return True
