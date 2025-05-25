# Setup script for MLOps cluster

Write-Host "MLOps Cluster Setup Başlıyor..." -ForegroundColor Green

# 1. Cluster'ı oluştur
Write-Host "`n1. Kind cluster oluşturuluyor..." -ForegroundColor Yellow
kind create cluster --config kind-config.yaml

# 2. Cluster'ı kontrol et
Write-Host "`n2. Cluster durumu kontrol ediliyor..." -ForegroundColor Yellow
kubectl get nodes

# 3. Namespace'leri oluştur
Write-Host "`n3. Namespace'ler oluşturuluyor..." -ForegroundColor Yellow
kubectl create namespace mlflow
kubectl create namespace airflow
kubectl create namespace monitoring
kubectl create namespace serving

# 4. MLflow'u deploy et
Write-Host "`n4. MLflow deploy ediliyor..." -ForegroundColor Yellow
kubectl apply -f kubernetes/deployments/mlflow-deployment.yaml
kubectl apply -f kubernetes/services/mlflow-service.yaml

# 5. Deployment durumunu kontrol et
Write-Host "`n5. Deployment durumu..." -ForegroundColor Yellow
kubectl get pods -n mlflow
kubectl get svc -n mlflow

Write-Host "`nSetup tamamlandı!" -ForegroundColor Green
Write-Host "MLflow'a erişmek için: http://localhost:30001" -ForegroundColor Cyan