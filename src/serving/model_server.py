from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime
import os

class DeliveryPredictionRequest(BaseModel):
    customer_id: str
    product_category: str
    order_value: float
    items_count: int
    weight_kg: float
    volume_cm3: float
    fragile: bool
    perishable: bool
    warehouse_id: str
    courier_id: str
    delivery_distance_km: float
    traffic_condition: str
    weather: str
    payment_method: str
    order_datetime: Optional[str] = None
    
class DeliveryPredictionResponse(BaseModel):
    predicted_delivery_hours: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    delivery_difficulty_score: float
    recommendations: List[str]

class BatchPredictionRequest(BaseModel):
    predictions: List[DeliveryPredictionRequest]

app = FastAPI(
    title="Lojistik Teslimat Süresi Tahmin API",
    description="ML tabanlı teslimat süresi tahmin servisi",
    version="1.0.0"
)

model_artifacts = None
feature_engineering = None

def load_model_artifacts(model_path="models/best_model.pkl"):
    """Model ve preprocessing nesnelerini yükle"""
    global model_artifacts
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    return model_artifacts

def prepare_features(request_data: Dict) -> pd.DataFrame:
    """API request'ini model için hazırla"""
    features = {
        'order_value': request_data['order_value'],
        'items_count': request_data['items_count'],
        'weight_kg': request_data['weight_kg'],
        'volume_cm3': request_data['volume_cm3'],
        'delivery_distance_km': request_data['delivery_distance_km'],
        'fragile': int(request_data['fragile']),
        'perishable': int(request_data['perishable'])
    }
    
    if request_data.get('order_datetime'):
        dt = pd.to_datetime(request_data['order_datetime'])
    else:
        dt = datetime.now()
    
    features['order_hour'] = dt.hour
    features['order_day_of_week'] = dt.dayofweek
    features['is_weekend'] = int(dt.dayofweek in [5, 6])
    features['is_peak_hour'] = int(dt.hour in [11, 12, 17, 18, 19])
    
    features['value_per_item'] = features['order_value'] / features['items_count']
    features['weight_per_item'] = features['weight_kg'] / features['items_count']
    features['volume_per_item'] = features['volume_cm3'] / features['items_count']
    
    features['complexity_score'] = (
        features['items_count'] * 0.3 +
        features['weight_kg'] * 0.2 +
        (features['fragile'] * 5) +
        (features['perishable'] * 10) +
        features['delivery_distance_km'] * 0.5
    )
    
    traffic_impact = {'low': 1, 'medium': 1.5, 'high': 2.5}
    weather_impact = {'sunny': 1, 'cloudy': 1.2, 'rainy': 1.8, 'snowy': 2.5}
    
    features['traffic_impact'] = traffic_impact.get(request_data['traffic_condition'], 1.5)
    features['weather_impact'] = weather_impact.get(request_data['weather'], 1.2)
    
    features['delivery_difficulty_score'] = (
        features['complexity_score'] * 
        features['traffic_impact'] * 
        features['weather_impact']
    )
    
    categorical_mappings = {
        'product_category': request_data['product_category'],
        'traffic_condition': request_data['traffic_condition'],
        'weather': request_data['weather'],
        'payment_method': request_data['payment_method']
    }
    
    df = pd.DataFrame([features])
    
    for feature_name in model_artifacts['feature_names']:
        if feature_name not in df.columns:
            for cat_var, value in categorical_mappings.items():
                if feature_name.startswith(f'{cat_var}_') and feature_name.endswith(value):
                    df[feature_name] = 1
                    break
            else:
                df[feature_name] = 0
    
    df = df[model_artifacts['feature_names']]
    
    return df

def generate_recommendations(prediction: float, difficulty_score: float) -> List[str]:
    """Tahmine göre öneriler üret"""
    recommendations = []
    
    if prediction > 48:
        recommendations.append("⚠️ Uzun teslimat süresi bekleniyor, müşteri bilgilendirilmeli")
    
    if difficulty_score > 50:
        recommendations.append("🚚 Deneyimli kurye ataması önerilir")
        recommendations.append("📦 Özel paketleme gerekebilir")
    
    if prediction < 24:
        recommendations.append("✅ Hızlı teslimat için uygun")
        recommendations.append("🎯 Premium teslimat seçeneği sunulabilir")
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    """API başlangıçında modeli yükle"""
    try:
        load_model_artifacts()
    except Exception as e:

@app.get("/")
async def root():
    """API ana endpoint'i"""
    return {
        "message": "Lojistik Teslimat Süresi Tahmin API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model_artifacts is not None
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {
        "status": "healthy",
        "model_loaded": model_artifacts is not None,
        "model_type": model_artifacts['model_type'] if model_artifacts else None
    }

@app.post("/predict", response_model=DeliveryPredictionResponse)
async def predict_delivery_time(request: DeliveryPredictionRequest):
    """Tekil teslimat süresi tahmini"""
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    
    try:
        features_df = prepare_features(request.dict())
        
        prediction = model_artifacts['model'].predict(features_df)[0]
        
        confidence_margin = prediction * 0.15  # %15 margin
        
        difficulty_score = features_df['delivery_difficulty_score'].values[0]
        
        recommendations = generate_recommendations(prediction, difficulty_score)
        
        return DeliveryPredictionResponse(
            predicted_delivery_hours=round(prediction, 2),
            confidence_interval_lower=round(prediction - confidence_margin, 2),
            confidence_interval_upper=round(prediction + confidence_margin, 2),
            delivery_difficulty_score=round(difficulty_score, 2),
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tahmin hatası: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """Toplu teslimat süresi tahmini"""
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    
    results = []
    for pred_request in request.predictions:
        try:
            result = await predict_delivery_time(pred_request)
            results.append({
                "customer_id": pred_request.customer_id,
                "prediction": result.dict()
            })
        except Exception as e:
            results.append({
                "customer_id": pred_request.customer_id,
                "error": str(e)
            })
    
    return {"results": results}

@app.get("/model_info")
async def model_info():
    """Model bilgileri"""
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    
    return {
        "model_type": model_artifacts['model_type'],
        "n_features": len(model_artifacts['feature_names']),
        "top_features": model_artifacts['feature_names'][:10]
    }

if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )