# RAGr

Multi-tenant RAG platform. Create knowledge-base-backed chat models, ingest content from text, URLs, files, or site
crawls, and query them through a streaming chat API.

Built with FastAPI, PostgreSQL + pgvector, Claude (Anthropic), and Voyage AI embeddings.

## Architecture

```
Client -> Cloudflare Tunnel -> FastAPI (uvicorn)
                                  |
                          PostgreSQL + pgvector
                                  |
                     Anthropic API  /  Voyage AI API
```

**Core flow:** Content is ingested, chunked, and embedded into pgvector. Chat queries embed the question, retrieve
relevant chunks via hybrid search (vector + keyword with RRF), and generate answers with Claude using the retrieved
context.

**Multi-tenancy:** Each "model" is an isolated tenant with its own knowledge base, system prompt, chat theme, budget
cap, API keys, and CORS origins. Models can use platform API keys or bring their own (encrypted at rest with Fernet).

## Features

- **Hybrid retrieval** -- vector similarity + full-text keyword search, merged with Reciprocal Rank Fusion
- **Reranking** -- optional Voyage AI reranker for improved precision
- **Streaming chat** -- SSE streaming with multi-turn conversation history
- **Content ingestion** -- plain text, URLs, file upload (PDF, HTML, TXT, Markdown), presigned R2 uploads, site crawling
- **Per-model budgets** -- monthly spend caps with automatic enforcement
- **Custom API keys** -- BYOK for Anthropic and Voyage, encrypted at rest
- **Per-model CORS** -- each model controls which origins can access it
- **Per-model API keys** -- scoped access tokens for programmatic use
- **System prompt tooling** -- AI-assisted prompt generation, version history, rollback
- **Admin analytics** -- per-model stats, daily breakdowns, top sources by retrieval count, conversation browser

## API

All model-scoped endpoints are under `/models/{slug}/`.

| Area         | Endpoints                                                    |
|--------------|--------------------------------------------------------------|
| **Models**   | CRUD, public info, chat theme                                |
| **Chat**     | `POST /models/{slug}/chat` (streaming or sync)               |
| **Sources**  | Text/URL/file/crawl ingestion, presigned upload, list/delete |
| **API Keys** | Create, list, revoke per-model keys                          |
| **Admin**    | Stats, daily analytics, conversations, system prompt history |
| **Health**   | `GET /healthz`, `GET /readyz`                                |

## Local Development

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (package manager)
- Docker (for PostgreSQL)

### Setup

```bash
# Clone and install
git clone https://github.com/cjwillis48/ragr.git
cd ragr
uv sync

# Start Postgres (with pgvector)
docker compose up -d postgres

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   ANTHROPIC_API_KEY, VOYAGE_API_KEY, CLERK_SECRET_KEY

# Run migrations
uv run alembic upgrade head

# Start the server
uv run uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. A pgweb UI is available at `http://localhost:8081` if you run
`docker compose up -d pgweb`.

### Environment Variables

| Variable               | Required | Default                                              | Description                                       |
|------------------------|----------|------------------------------------------------------|---------------------------------------------------|
| `DATABASE_URL`         | No       | `postgresql+asyncpg://ragr:ragr@localhost:5432/ragr` | Async PostgreSQL connection string                |
| `ANTHROPIC_API_KEY`    | Yes      | --                                                   | Platform Anthropic API key                        |
| `VOYAGE_API_KEY`       | Yes      | --                                                   | Platform Voyage AI API key                        |
| `CLERK_SECRET_KEY`     | Yes      | --                                                   | Clerk authentication secret                       |
| `ENCRYPTION_KEY`       | No       | --                                                   | Fernet key for encrypting custom API keys         |
| `CONSOLE_ORIGINS`      | No       | `["http://localhost:5173"]`                          | Allowed origins for the admin console             |
| `SUPERUSER_ID`         | No       | --                                                   | Clerk user ID with read access to all models      |
| `R2_ACCOUNT_ID`        | No       | --                                                   | Cloudflare R2 account (enables presigned uploads) |
| `R2_ACCESS_KEY_ID`     | No       | --                                                   | R2 access key                                     |
| `R2_SECRET_ACCESS_KEY` | No       | --                                                   | R2 secret key                                     |
| `R2_BUCKET_NAME`       | No       | `ragr-uploads`                                       | R2 bucket name                                    |

## Deployment

RAGr deploys to Kubernetes via GitHub Actions + Argo CD:

1. Push to `main` triggers a GH Actions build (Docker image to GHCR)
2. The workflow updates the image tag in `k8s/ragr/deployment.yaml` and commits
3. Argo CD syncs the manifests to the cluster

Secrets are managed with Bitnami Sealed Secrets. Network policies enforce default-deny with explicit allowlists. The app
runs as non-root with a read-only filesystem and dropped capabilities.

## Tech Stack

| Layer      | Technology                          |
|------------|-------------------------------------|
| API        | FastAPI, uvicorn                    |
| Database   | PostgreSQL 16 + pgvector            |
| Embeddings | Voyage AI (voyage-4-lite)           |
| Generation | Anthropic Claude                    |
| Reranking  | Voyage AI (rerank-2.5-lite)         |
| Auth       | Clerk                               |
| Storage    | Cloudflare R2 (optional)            |
| Infra      | Kubernetes, GitHub Actions, Argo CD |
| Secrets    | Bitnami Sealed Secrets              |

## License

MIT
