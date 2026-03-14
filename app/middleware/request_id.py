import logging
import time
import uuid
from contextvars import ContextVar

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")

_access_logger = logging.getLogger("ragr.access")

# Paths excluded from access logging (K8s probes).
_SILENT_PATHS = {"/healthz", "/readyz"}


class RequestIdMiddleware:
    """Pure ASGI middleware — safe for streaming responses.

    Reads X-Request-ID from the incoming request, or generates a UUID.
    Sets REQUEST_ID_CTX so all log lines in the same request carry the ID.
    Echoes the ID back in the X-Request-ID response header.
    Also logs each HTTP request with method, path, status, and duration.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        request_id = headers.get(b"x-request-id", b"").decode().strip() or str(uuid.uuid4())
        REQUEST_ID_CTX.set(request_id)

        status_code = 0
        start = time.monotonic()

        async def send_with_header(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                new_headers = list(message.get("headers", []))
                new_headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": new_headers}
            await send(message)

        await self.app(scope, receive, send_with_header)

        path = scope.get("path", "")
        if path not in _SILENT_PATHS:
            method = scope.get("method", "?")
            qs = scope.get("query_string", b"").decode()
            full_path = f"{path}?{qs}" if qs else path
            duration_ms = (time.monotonic() - start) * 1000
            _access_logger.info(
                '%s %s %d (%.0fms)', method, full_path, status_code, duration_ms,
            )


class RequestIdFilter(logging.Filter):
    """Injects request_id from context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get("-")  # type: ignore[attr-defined]
        return True
