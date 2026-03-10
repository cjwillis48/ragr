import logging
import uuid
from contextvars import ContextVar

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdMiddleware:
    """Pure ASGI middleware — safe for streaming responses.

    Reads X-Request-ID from the incoming request, or generates a UUID.
    Sets REQUEST_ID_CTX so all log lines in the same request carry the ID.
    Echoes the ID back in the X-Request-ID response header.
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

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                new_headers = list(message.get("headers", []))
                new_headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": new_headers}
            await send(message)

        await self.app(scope, receive, send_with_header)


class RequestIdFilter(logging.Filter):
    """Injects request_id from context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get("-")  # type: ignore[attr-defined]
        return True
