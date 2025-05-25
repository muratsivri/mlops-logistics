from prometheus_client import Counter, Histogram, Gauge, generate_latest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
import json
import os
from typing import Dict, List
import sqlite3

# Prometheus metrikleri
prediction_counter = Counter('delivery_predictions_total', 'Toplam tahmin sayısı')
prediction_latency = Histogram('prediction_latency_seconds', 'Tahmin süresi')
prediction_error = Histogram('prediction_error_hours', 'Tahmin hatası (saat)')
model_accuracy = Gauge('model_accuracy_mae', 'Model MAE skoru')
data_drift_score = Gauge('data_drift_score', 'Veri kayması skoru')

class MonitoringService:
    """Model monitoring ve metrik toplama servisi"""
    
    def __init__(self, db_path="monitoring/metrics.db", mlflow_uri="http://localhost:5000"):
        self.db_path = db_path
        mlflow.set_tracking_uri(mlflow_uri)
        self.init_database()
        
    def init_database(self):
        """Monitoring veritabanını başlat"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tahmin logları tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                customer_id TEXT,
                predicted_hours REAL,
                actual_hours REAL,
                prediction_error REAL,
                features TEXT,
                model_version TEXT
            )
        ''')
        
        # Metrikler tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                tags TEXT
            )
        ''')
        
        # Data drift tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_drift (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feature_name TEXT,
                drift_score REAL,
                is_drifted BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, customer_id: str, predicted_hours: float, 
                      features: Dict, actual_hours: float = None, model_version: str = "1.0"):
        """Tahmin logu kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prediction_error = None
        if actual_hours:
            prediction_error = abs(predicted_hours - actual_hours)
            prediction_error_metric.observe(prediction_error)
        
        cursor.execute('''
            INSERT INTO predictions (customer_id, predicted_hours, actual_hours, 
                                   prediction_error, features, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (customer_id, predicted_hours, actual_hours, prediction_error, 
              json.dumps(features), model_version))
        
        conn.commit()
        conn.close()
        
        # Prometheus metriklerini güncelle
        prediction_counter.inc()
    
    def log_metric(self, metric_name: str, metric_value: float, tags: Dict = None):
        """Genel metrik kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (metric_name, metric_value, tags)
            VALUES (?, ?, ?)
        ''', (metric_name, metric_value, json.dumps(tags or {})))
        
        conn.commit()
        conn.close()
    
    def calculate_model_performance(self, last_n_days: int = 7):
        """Model performansını hesapla"""
        conn = sqlite3.connect(self.db_path)
        
        # Son N gündeki tahminleri al
        query = '''
            SELECT predicted_hours, actual_hours, prediction_error
            FROM predictions
            WHERE actual_hours IS NOT NULL
            AND timestamp >= datetime('now', '-{} days')
        '''.format(last_n_days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return None
        
        # Metrikler hesapla
        mae = np.mean(df['prediction_error'])
        rmse = np.sqrt(np.mean(df['prediction_error'] ** 2))
        mape = np.mean(np.abs(df['prediction_error'] / df['actual_hours'])) * 100
        
        # Prometheus metriklerini güncelle
        model_accuracy.set(mae)
        
        # MLflow'a logla
        with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_metrics({
                "monitoring_mae": mae,
                "monitoring_rmse": rmse,
                "monitoring_mape": mape,
                "monitoring_sample_size": len(df)
            })
        
        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "sample_size": len(df)
        }
    
    def detect_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                         feature_columns: List[str]):
        """Veri kayması tespiti"""
        # Evidently report oluştur
        column_mapping = ColumnMapping(
            target='actual_delivery_hours',
            numerical_features=feature_columns
        )
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data, 
                  column_mapping=column_mapping)
        
        # Sonuçları al
        result = report.as_dict()
        
        # Drift skorlarını kaydet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for feature in feature_columns:
            if feature in result['metrics'][0]['result']['drift_by_columns']:
                drift_info = result['metrics'][0]['result']['drift_by_columns'][feature]
                drift_score = drift_info.get('drift_score', 0)
                is_drifted = drift_info.get('drift_detected', False)
                
                cursor.execute('''
                    INSERT INTO data_drift (feature_name, drift_score, is_drifted)
                    VALUES (?, ?, ?)
                ''', (feature, drift_score, is_drifted))
                
                # Prometheus metriği
                if is_drifted:
                    data_drift_score.set(drift_score)
        
        conn.commit()
        conn.close()
        
        return result
    
    def generate_monitoring_report(self, output_path: str = "monitoring/reports"):
        """Detaylı monitoring raporu oluştur"""
        os.makedirs(output_path, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Son 30 günlük tahmin performansı
        performance_query = '''
            SELECT DATE(timestamp) as date,
                   COUNT(*) as prediction_count,
                   AVG(prediction_error) as avg_error,
                   MAX(prediction_error) as max_error,
                   MIN(prediction_error) as min_error
            FROM predictions
            WHERE actual_hours IS NOT NULL
            AND timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        '''
        
        performance_df = pd.read_sql_query(performance_query, conn)
        
        # Data drift özeti
        drift_query = '''
            SELECT feature_name,
                   AVG(drift_score) as avg_drift_score,
                   SUM(is_drifted) as drift_count,
                   COUNT(*) as total_checks
            FROM data_drift
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY feature_name
            ORDER BY avg_drift_score DESC
        '''
        
        drift_df = pd.read_sql_query(drift_query, conn)
        
        conn.close()
        
        # Rapor oluştur
        report = {
            "generated_at": datetime.now().isoformat(),
            "performance_summary": {
                "last_30_days_avg_error": performance_df['avg_error'].mean(),
                "last_7_days_avg_error": performance_df.head(7)['avg_error'].mean(),
                "total_predictions": performance_df['prediction_count'].sum(),
                "worst_day": performance_df.loc[performance_df['avg_error'].idxmax()].to_dict() if len(performance_df) > 0 else None
            },
            "drift_summary": {
                "features_with_drift": drift_df[drift_df['drift_count'] > 0]['feature_name'].tolist(),
                "avg_drift_scores": drift_df.set_index('feature_name')['avg_drift_score'].to_dict()
            },
            "daily_performance": performance_df.to_dict('records'),
            "drift_details": drift_df.to_dict('records')
        }
        
        # JSON olarak kaydet
        report_path = os.path.join(output_path, f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Monitoring raporu oluşturuldu: {report_path}")
        
        return report
    
    def get_prometheus_metrics(self):
        """Prometheus metrikleri döndür"""
        return generate_latest()
    
    def alert_check(self):
        """Alert kontrolü"""
        alerts = []
        
        # Model performans kontrolü
        performance = self.calculate_model_performance(last_n_days=1)
        if performance and performance['mae'] > 5:  # 5 saatten fazla ortalama hata
            alerts.append({
                "type": "performance_degradation",
                "severity": "high",
                "message": f"Model MAE değeri yüksek: {performance['mae']:.2f} saat",
                "timestamp": datetime.now().isoformat()
            })
        
        # Tahmin sayısı kontrolü
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM predictions
            WHERE timestamp >= datetime('now', '-1 hour')
        ''')
        
        hourly_predictions = cursor.fetchone()[0]
        
        if hourly_predictions < 10:  # Saatte 10'dan az tahmin
            alerts.append({
                "type": "low_prediction_volume",
                "severity": "medium",
                "message": f"Son 1 saatte düşük tahmin sayısı: {hourly_predictions}",
                "timestamp": datetime.now().isoformat()
            })
        
        conn.close()
        
        return alerts

class ModelRegistry:
    """Model versiyonlama ve registry"""
    
    def __init__(self, mlflow_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(mlflow_uri)
        self.registry_path = "models/registry"
        os.makedirs(self.registry_path, exist_ok=True)
    
    def register_model(self, model_path: str, model_name: str, version: str, 
                      metrics: Dict, description: str = ""):
        """Modeli kaydet"""
        # MLflow model registry
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model_path,
                artifact_path="model",
                registered_model_name=model_name
            )
        
        # Local registry
        registry_info = {
            "model_name": model_name,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "description": description,
            "model_path": model_path
        }
        
        registry_file = os.path.join(self.registry_path, f"{model_name}_v{version}.json")
        with open(registry_file, 'w') as f:
            json.dump(registry_info, f, indent=2)
        
        print(f"✅ Model kaydedildi: {model_name} v{version}")
    
    def get_latest_model(self, model_name: str):
        """En son model versiyonunu al"""
        # Registry dosyalarını kontrol et
        registry_files = [f for f in os.listdir(self.registry_path) 
                         if f.startswith(model_name) and f.endswith('.json')]
        
        if not registry_files:
            return None
        
        # En son versiyonu bul
        latest_file = sorted(registry_files)[-1]
        
        with open(os.path.join(self.registry_path, latest_file), 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    # Monitoring servisi örnek kullanım
    monitor = MonitoringService()
    
    # Örnek tahmin logu
    monitor.log_prediction(
        customer_id="CUST_00001",
        predicted_hours=24.5,
        features={"distance": 10, "items": 3},
        actual_hours=26.0
    )
    
    # Performans hesapla
    performance = monitor.calculate_model_performance()
    if performance:
        print(f"Model Performansı: {performance}")
    
    # Monitoring raporu oluştur
    report = monitor.generate_monitoring_report()
    
    # Alert kontrolü
    alerts = monitor.alert_check()
    if alerts:
        print(f"⚠️ Aktif alertler: {alerts}")