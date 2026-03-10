import logging
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from app.models.rag_model import RagModel

logger = logging.getLogger("ragr.cors")

# slug -> allowed origins, updated at startup and on model CRUD
_origins_by_slug: dict[str, list[str]] = {}

# Matches /models/{slug}/... paths
_SLUG_RE = re.compile(r"^/models/([a-z0-9][a-z0-9-]*)/")


class DynamicCORSMiddleware:
    """Per-model CORS middleware.

    For /models/{slug}/... routes, only allows origins configured on that model.
    Non-model routes (healthz, readyz, etc.) pass through with no CORS headers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._cors = CORSMiddleware(
            app=app,
            allow_origins=[],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        match = _SLUG_RE.match(path)
        if match:
            slug = match.group(1)
            self._cors.allow_origins = _origins_by_slug.get(slug, [])
        else:
            # Non-model routes: allow any origin that's configured on at least one model
            self._cors.allow_origins = list({o for origins in _origins_by_slug.values() for o in origins})

        await self._cors(scope, receive, send)


async def sync_origins(session: AsyncSession) -> None:
    """Rebuild the per-model CORS origin map from all active models."""
    result = await session.execute(
        select(RagModel.slug, RagModel.allowed_origins).where(RagModel.is_active == True)  # noqa: E712
    )
    new_map: dict[str, list[str]] = {}
    for slug, model_origins in result.all():
        new_map[slug] = list(model_origins) if model_origins else []

    _origins_by_slug.clear()
    _origins_by_slug.update(new_map)
    logger.info("CORS origins synced: %s", {k: v for k, v in _origins_by_slug.items() if v})
