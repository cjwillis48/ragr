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

    Caches CORSMiddleware instances by frozen origin set to avoid per-request construction.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._cache: dict[frozenset[str], CORSMiddleware] = {}

    def _get_cors(self, origins: list[str]) -> CORSMiddleware:
        key = frozenset(origins)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cors = CORSMiddleware(
            app=self._app,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._cache[key] = cors
        return cors

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        from app.config import settings  # lazy: avoids circular import (cors → main → cors)

        path = scope.get("path", "")
        match = _SLUG_RE.match(path)
        if match:
            slug = match.group(1)
            model_origins = _origins_by_slug.get(slug, [])
            origins = list(set(model_origins + settings.console_origins))
        else:
            origins = list(settings.console_origins)

        cors = self._get_cors(origins)
        await cors(scope, receive, send)


async def sync_origins(session: AsyncSession) -> None:
    """Rebuild the per-model CORS origin map from all active models."""
    result = await session.execute(
        select(RagModel.slug, RagModel.allowed_origins).where(
            RagModel.is_active == True,  # noqa: E712
            RagModel.deleted_at.is_(None),
        )
    )
    new_map: dict[str, list[str]] = {}
    for slug, model_origins in result.all():
        new_map[slug] = list(model_origins) if model_origins else []

    _origins_by_slug.clear()
    _origins_by_slug.update(new_map)
    logger.info("CORS origins synced: %s", {k: v for k, v in _origins_by_slug.items() if v})
