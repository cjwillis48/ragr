NAMESPACE := ragr
TEST_DATABASE_URL := postgresql+asyncpg://ragr:ragr@localhost:5432/ragr_test
EVAL_DATABASE_URL ?= postgresql+asyncpg://ragr:ragr@localhost:5432/ragr_eval
EVAL_MODEL ?= chatlie

.PHONY: restart logs status enter-pg enter-ragr db-export db-import test test-unit test-cov test-integration eval

restart:
	kubectl rollout restart deployment/ragr -n $(NAMESPACE)

logs:
	kubectl logs -n $(NAMESPACE) -l app=ragr -c ragr -f

status:
	kubectl get pods -n $(NAMESPACE)

enter-pg:
	kubectl exec -it -n $(NAMESPACE) postgres-0 -- psql -U ragr -d ragr

enter-ragr:
	kubectl exec -it -n $(NAMESPACE) deployment/ragr -- /bin/bash

db-export:
	kubectl exec -n $(NAMESPACE) postgres-0 -- pg_dump -U ragr -d ragr > backup.sql
	@echo "Exported to backup.sql ($$(wc -c < backup.sql) bytes)"

db-import:
	kubectl exec -i -n $(NAMESPACE) postgres-0 -- psql -U ragr -d ragr < backup.sql
	@echo "Import complete"

test:
	uv run pytest -x -q

test-unit:
	uv run pytest tests/unit -x -q

test-cov:
	uv run pytest --cov=app --cov-report=term-missing

# Integration tests need a real Postgres. docker-compose only creates the `ragr`
# database, so create `ragr_test` (and its pgvector extension) on first run.
#
# ragr_test survives between runs, which means it keeps whichever alembic
# revision the last branch stamped it with. Switch to a branch that doesn't
# contain that revision and every test errors during fixture setup with
# "Can't locate revision identified by ...", which points nowhere near the
# actual cause. So: if the stamped revision isn't in this branch's migrations,
# drop the database and start clean.
test-integration:
	docker compose up -d postgres
	@until docker compose exec -T postgres pg_isready -U ragr -q; do sleep 1; done
	@stamped=$$(docker compose exec -T postgres psql -U ragr -d ragr_test -tAc \
		"SELECT version_num FROM alembic_version" 2>/dev/null | tr -d '[:space:]'); \
	if [ -n "$$stamped" ] && ! grep -rqs "revision = [\"']$$stamped[\"']" migrations/versions/; then \
		echo "ragr_test is stamped at $$stamped, which this branch doesn't have — recreating."; \
		docker compose exec -T postgres dropdb -U ragr ragr_test; \
	fi
	@docker compose exec -T postgres psql -U ragr -d ragr -tAc \
		"SELECT 1 FROM pg_database WHERE datname='ragr_test'" | grep -q 1 \
		|| docker compose exec -T postgres createdb -U ragr ragr_test
	@docker compose exec -T postgres psql -U ragr -d ragr_test -qc "CREATE EXTENSION IF NOT EXISTS vector;"
	DATABASE_URL="$(TEST_DATABASE_URL)" uv run pytest tests/integration -q

# Score retrieval against a golden set. Worth running before shipping anything
# that touches retrieval — the goldens are recorded prod failures, so a drop
# means one of them came back.
#
#   make eval
#   make eval EVAL_MODEL=hades-bot
#   make eval ARGS="--no-context --json runs/before.json"
#
# Needs a corpus in EVAL_DATABASE_URL and VOYAGE_API_KEY (queries are embedded
# and reranked live). Point it at the owner role, not ragr_app: this resolves a
# model without going through the request path, so it never binds a tenant.
eval:
	@DATABASE_URL="$(EVAL_DATABASE_URL)" uv run python tests/eval/eval_retrieval.py \
		--model $(EVAL_MODEL) $(ARGS)
