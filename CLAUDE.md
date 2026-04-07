# RAGr — Project Guidelines

## Code Style

- **Imports at the top of the file.** Only use local/lazy imports when necessary to avoid circular imports or defer heavy dependencies. Add a `# lazy: <reason>` comment when a local import is warranted.
- **No unnecessary abstractions.** Don't create helpers, utilities, or wrappers for one-time operations.
- **Keep functions focused.** Extract private functions when a block of logic is reusable or when it improves readability, but not preemptively.

## Testing

- Unit tests: `tests/unit/` — no DB, no network, fast.
- Integration tests: `tests/integration/` — real PostgreSQL, mocked external services.
- Run unit tests: `uv run pytest tests/unit/`
- Run integration tests: `DATABASE_URL="postgresql+asyncpg://ragr:ragr@localhost:5432/ragr_test" uv run pytest tests/integration/`

## Database

- Migrations via alembic: `migrations/versions/`
- PostgreSQL + pgvector required
- Local dev: `docker compose up -d postgres`
