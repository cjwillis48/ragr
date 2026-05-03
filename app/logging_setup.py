"""Structured JSON logging configuration.

Import this module to configure logging for any process (API server or worker).
"""

import json
import logging
from datetime import UTC, datetime

from app.middleware.log_context import LogContextFilter

_BUILTIN_LOG_ATTRS = logging.LogRecord("", 0, "", 0, None, None, None).__dict__.keys()


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter.

    Merges standard fields (timestamp, level, logger, request_id, message)
    with any extra fields passed via the ``extra`` dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _BUILTIN_LOG_ATTRS and key not in entry:
                entry[key] = val
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    """Set up structured JSON logging. Safe to call multiple times (idempotent)."""
    if getattr(configure_logging, "_done", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    handler.addFilter(LogContextFilter())
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    # Remove default handlers added before our setup
    for h in logging.root.handlers[:-1]:
        logging.root.removeHandler(h)
    configure_logging._done = True
