import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.cors import DynamicCORSMiddleware, sync_origins
from app.database import async_session, engine, get_session
from app.middleware.request_id import RequestIdFilter, RequestIdMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Inject request_id into every log record
_request_id_filter = RequestIdFilter()
for _handler in logging.root.handlers:
    _handler.addFilter(_request_id_filter)

logger = logging.getLogger("ragr")


class _HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/healthz" not in msg and "/readyz" not in msg


logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.ragr_api_key:
        raise RuntimeError("RAGR_API_KEY must be set")
    logger.info("RAGr starting up")
    async with async_session() as session:
        await sync_origins(session)
    yield
    logger.info("RAGr shutting down")
    await engine.dispose()


app = FastAPI(title="RAGr", version="0.1.0", lifespan=lifespan)

app.add_middleware(RequestIdMiddleware)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Strip raw input from validation errors to prevent input reflection."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": e.get("loc", []), "msg": e.get("msg", ""), "type": e.get("type", "")}
                for e in exc.errors()
            ]
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

app.add_middleware(DynamicCORSMiddleware)


@app.get("/healthz")
async def healthz():
    """K8s liveness probe. Always 200 if the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(session: AsyncSession = Depends(get_session)):
    """K8s readiness probe. 200 if the app can reach the database."""
    try:
        await session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        logger.error("Unable to connect to database: %s", settings.database_url, exc_info=True)
        return JSONResponse(status_code=503, content={"status": "unavailable"})



# Register routers
from app.api.models import router as models_router  # noqa: E402
from app.api.chat import router as chat_router  # noqa: E402

from app.api.admin import router as admin_router  # noqa: E402
from app.api.api_keys import router as api_keys_router  # noqa: E402
from app.api.sources import router as sources_router  # noqa: E402

app.include_router(models_router)
app.include_router(chat_router)

app.include_router(admin_router)
app.include_router(api_keys_router)
app.include_router(sources_router)
