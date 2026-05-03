# RAGr — Project Guidelines

## What this is

Multi-tenant RAG platform. Each "model" is an isolated tenant with its own knowledge base, system prompt, budget cap, API keys, and CORS origins. FastAPI + Postgres/pgvector, Anthropic Claude for generation, Voyage AI for embeddings + reranking. Content is ingested, chunked, embedded, and retrieved via hybrid search (vector + keyword with RRF) before being passed to Claude.

The admin frontend (`ragr-console`) lives in a sibling directory, **not** inside this repo.

## Layout

```
app/
  main.py            FastAPI app (API process)
  worker.py          Background ingestion worker (separate process)
  api/               Route modules: models, chat, sources, api_keys, admin
  services/          Domain logic: ingest, chunker, embedder, retrieval,
                     generation, reranker, crawler, html, r2, crypto, ...
  models/            SQLAlchemy ORM models
  schemas/           Pydantic request/response schemas
  middleware/        CORS (dynamic, per-model), request_id
migrations/versions/ Alembic migrations
k8s/                 Manifests: ragr, ragr-worker, postgres, secrets
scripts/             One-off utilities (chatlie, sponsorbot, debug_retrieval)
tests/{unit,integration}
```

## Architecture notes

- **Two processes:** the API (`app.main:app` under uvicorn) and the worker (`python -m app.worker`). They share the database but never call each other directly.
- **Job queue:** ingestion is async via the `ingestion_jobs` table, claimed with `SELECT … FOR UPDATE SKIP LOCKED` for safe multi-worker concurrency.
- **Multi-tenancy:** every tenant-scoped query must filter by `model_id`. Endpoints under `/models/{slug}/...` resolve the slug to a `RagModel` and scope from there.
- **BYOK secrets:** custom Anthropic/Voyage keys are encrypted at rest with Fernet (`ENCRYPTION_KEY`). Don't log or return the plaintext.
- **Per-model CORS:** allowed origins are stored in the DB and synced into `DynamicCORSMiddleware` at startup and on model changes.
- **Deploy:** push to `main` → GH Actions builds image → workflow bumps tag in `k8s/ragr/deployment.yaml` → Argo CD syncs.
- **HNSW vector index** is intentionally deferred until scale requires it (~50k chunks per model). Don't add it preemptively.

## Code style

- **Imports at the top of the file.** Local/lazy imports only to break circulars or defer heavy deps. Add a `# lazy: <reason>` comment when warranted.
- **No unnecessary abstractions.** Don't create helpers, utilities, or wrappers for one-time operations.
- **Keep functions focused.** Extract a private function when logic is reused or readability genuinely improves — not preemptively.
- **No comments explaining what well-named code already says.** Comment the non-obvious *why*.

## Testing

- `tests/unit/` — no DB, no network, fast.
- `tests/integration/` — real PostgreSQL, mocked external services (Anthropic, Voyage, R2).
- `asyncio_mode = "auto"` — async tests don't need `@pytest.mark.asyncio`.
- Mark tests >1s with `@pytest.mark.slow`.

```bash
make test                    # full suite, fail-fast, quiet
make test-unit               # unit only
make test-cov                # with coverage report

# integration (needs the test DB up):
DATABASE_URL="postgresql+asyncpg://ragr:ragr@localhost:5432/ragr_test" \
  uv run pytest tests/integration/
```

## Common commands

```bash
uv sync                                          # install deps
docker compose up -d postgres                    # start local PG (+ pgvector)
uv run uvicorn app.main:app --reload             # run API
uv run python -m app.worker                      # run ingestion worker
uv run alembic upgrade head                      # apply migrations
uv run alembic revision --autogenerate -m "..."  # new migration
make logs / make enter-pg / make enter-ragr      # k8s shortcuts
```

Environment variables are documented in `README.md`; copy `.env.example` to `.env` for local dev.

## Database

- PostgreSQL 16 + pgvector. Async via `asyncpg` + SQLAlchemy 2.x async.
- All schema changes go through Alembic — never hand-edit the DB.
- Sessions are obtained via the `get_session` dependency in API code, or `db.async_session()` in the worker / scripts.