FROM python:3.9-slim

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# MLflow ve bağımlılıkları yükle
RUN pip install --upgrade pip && \
    pip install mlflow[extras] gunicorn

# Çalışma dizini
WORKDIR /app

# mlruns klasörü oluştur (artifact'lar için)
RUN mkdir -p /mlruns

# Ortam değişkenleri ayarla (isteğe bağlı)
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000

CMD ["mlflow", "server"]
