"""Logging context: ContextVars and filter for structured log enrichment.

REQUEST_ID_CTX is set by RequestIdMiddleware.
MODEL_ID_CTX is set by model-resolving dependencies (get_model_by_slug, etc.).

The LogContextFilter injects both into every log record automatically.
"""

import logging
from contextvars import ContextVar

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")
MODEL_ID_CTX: ContextVar[int | None] = ContextVar("model_id", default=None)


class LogContextFilter(logging.Filter):
    """Injects request_id and model_id from context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get("-")  # type: ignore[attr-defined]
        model_id = MODEL_ID_CTX.get(None)
        if model_id is not None:
            record.model_id = model_id  # type: ignore[attr-defined]
        return True
