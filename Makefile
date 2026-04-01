NAMESPACE := ragr

.PHONY: restart logs status enter-pg enter-ragr db-export db-import test test-unit test-cov

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
