#!/usr/bin/env bash
# Build the image and deploy API + monitoring to a local cluster.
# Works with Minikube or Docker Desktop Kubernetes.  Usage: bash scripts/deploy_k8s.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=heart-api:1.0.0

docker build -t "$IMAGE" .
if command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
  minikube image load "$IMAGE"
fi

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl -n heart-mlops create configmap grafana-dashboards \
  --from-file=heart_api.json=monitoring/grafana/dashboards/heart_api.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring/monitoring.yaml

kubectl -n heart-mlops rollout status deployment/heart-api --timeout=180s
kubectl -n heart-mlops get pods,svc,hpa
echo
echo "API:      minikube: run 'minikube tunnel' then curl http://127.0.0.1/health"
echo "          Docker Desktop: curl http://localhost/health"
echo "Grafana:  kubectl -n heart-mlops port-forward svc/grafana 3000:3000  (admin/admin)"
echo "Prometheus: kubectl -n heart-mlops port-forward svc/prometheus 9090:9090"
