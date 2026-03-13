IMAGE := ghcr.io/cjwillis48/ragr
NAMESPACE := ragr
CERT := k8s/secrets/sealed-secrets-pub.pem

.PHONY: build push build-push deploy restart logs status enter-pg enter-ragr seal-secret

build:
	docker buildx build --platform linux/arm64 -t $(IMAGE):dev .

push:
	docker push $(IMAGE):dev

build-push: build push

deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/secrets/ragr-secrets.sealed.yaml
	kubectl apply -f k8s/secrets/ghcr-pull-secret.sealed.yaml
	kubectl apply -f k8s/postgres/
	kubectl apply -f k8s/ragr/

seal-secret:
	@test -n "$(IN)" || (echo "Usage: make seal-secret IN=/tmp/plain.yml OUT=k8s/secrets/output.sealed.yaml" && exit 1)
	@test -n "$(OUT)" || (echo "Usage: make seal-secret IN=/tmp/plain.yml OUT=k8s/secrets/output.sealed.yaml" && exit 1)
	kubeseal --format yaml --cert $(CERT) < $(IN) > $(OUT)

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
