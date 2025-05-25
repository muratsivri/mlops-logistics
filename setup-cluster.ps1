

kind create cluster --config kind-config.yaml

kubectl get nodes

kubectl create namespace mlflow
kubectl create namespace airflow
kubectl create namespace monitoring
kubectl create namespace serving

kubectl apply -f kubernetes/services/mlflow-service.yaml

kubectl get pods -n mlflow
kubectl get svc -n mlflow

