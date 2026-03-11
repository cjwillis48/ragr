IMAGE := ghcr.io/cjwillis48/ragr:latest
NAMESPACE := ragr

.PHONY: build push build-push deploy restart logs status enter-pg enter-ragr secrets edit-secrets

build:
	docker buildx build --platform linux/arm64 -t $(IMAGE) .

push:
	docker push $(IMAGE)

build-push: build push

deploy:
	kubectl apply -f k8s/namespace.yaml
	sops -d k8s/secrets/ragr-secrets.sops.yml | kubectl apply -f -
	kubectl apply -f k8s/postgres/
	kubectl apply -f k8s/ragr/

secrets:
	sops -d k8s/secrets/ragr-secrets.sops.yml | kubectl apply -f -

edit-secrets:
	sops k8s/secrets/ragr-secrets.sops.yml

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
