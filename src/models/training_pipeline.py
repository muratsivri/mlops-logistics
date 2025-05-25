import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Teslimat süresi tahmin modeli eğitimi"""
    
    def __init__(self, mlflow_tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')
        
    def load_features(self, features_dir="data/features"):
        """Feature'ları ve target'ı yükle"""
        print("📂 Özellikler yükleniyor...")
        
        features = pd.read_csv(f"{features_dir}/features.csv")
        target = pd.read_csv(f"{features_dir}/target.csv")['target']
        
        # Preprocessing nesnelerini yükle
        with open(f"{features_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{features_dir}/label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(f"{features_dir}/feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"✅ {len(features)} örnek, {len(self.feature_names)} özellik yüklendi")
        
        return features, target
    
    def split_data(self, features, target, test_size=0.2, val_size=0.1, random_state=42):
        """Train, validation ve test setlerine ayır"""
        print("🔄 Veri bölünüyor...")
        
        # İlk olarak train+val ve test'e ayır
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Train ve validation'a ayır
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(self, model, X, y, dataset_name=""):
        """Model performansını değerlendir"""
        predictions = model.predict(X)
        
        metrics = {
            f'mse_{dataset_name}': mean_squared_error(y, predictions),
            f'rmse_{dataset_name}': np.sqrt(mean_squared_error(y, predictions)),
            f'mae_{dataset_name}': mean_absolute_error(y, predictions),
            f'r2_{dataset_name}': r2_score(y, predictions),
            f'mape_{dataset_name}': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        return metrics, predictions
    
    def train_linear_models(self, X_train, y_train, X_val, y_val):
        """Linear modelleri eğit"""
        print("\n📈 Linear modeller eğitiliyor...")
        
        # Linear Regression
        with mlflow.start_run(run_name="linear_regression", nested=True):
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Değerlendirme
            train_metrics, _ = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(model, X_val, y_val, "val")
            
            # MLflow'a logla
            mlflow.log_params({"model_type": "LinearRegression"})
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(model, "model")
            
            self.models['linear'] = model
            print(f"  Linear Regression - Val RMSE: {val_metrics['rmse_val']:.3f}")
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = model
        
        # Ridge Regression
        with mlflow.start_run(run_name="ridge_regression", nested=True):
            param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
            model = Ridge()
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Değerlendirme
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            # MLflow'a logla
            mlflow.log_params({
                "model_type": "Ridge",
                "best_alpha": grid_search.best_params_['alpha']
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(best_model, "model")
            
            self.models['ridge'] = best_model
            print(f"  Ridge Regression - Val RMSE: {val_metrics['rmse_val']:.3f}")
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
    
    def train_tree_models(self, X_train, y_train, X_val, y_val):
        """Tree-based modelleri eğit"""
        print("\n🌳 Tree-based modeller eğitiliyor...")
        
        # Random Forest
        with mlflow.start_run(run_name="random_forest", nested=True):
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Değerlendirme
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            # MLflow'a logla
            mlflow.log_params({
                "model_type": "RandomForest",
                **grid_search.best_params_
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
            
            self.models['random_forest'] = best_model
            print(f"  Random Forest - Val RMSE: {val_metrics['rmse_val']:.3f}")
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
        
        # Gradient Boosting
        with mlflow.start_run(run_name="gradient_boosting", nested=True):
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
            
            model = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Değerlendirme
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            # MLflow'a logla
            mlflow.log_params({
                "model_type": "GradientBoosting",
                **grid_search.best_params_
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(best_model, "model")
            
            self.models['gradient_boosting'] = best_model
            print(f"  Gradient Boosting - Val RMSE: {val_metrics['rmse_val']:.3f}")
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost modelini eğit"""
        print("\n🚀 XGBoost eğitiliyor...")
        
        with mlflow.start_run(run_name="xgboost", nested=True):
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            model = XGBRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Değerlendirme
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            # MLflow'a logla
            mlflow.log_params({
                "model_type": "XGBoost",
                **grid_search.best_params_
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.xgboost.log_model(best_model, "model")
            
            self.models['xgboost'] = best_model
            print(f"  XGBoost - Val RMSE: {val_metrics['rmse_val']:.3f}")
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
    
    def final_evaluation(self, X_test, y_test):
        """En iyi model ile final değerlendirme"""
        print("\n🏆 En iyi model ile test değerlendirmesi...")
        
        test_metrics, predictions = self.evaluate_model(
            self.best_model, X_test, y_test, "test"
        )
        
        # Tahmin analizi
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions,
            'error': y_test - predictions,
            'error_percentage': np.abs((y_test - predictions) / y_test) * 100
        })
        
        print(f"\n📊 Test Sonuçları:")
        print(f"  - RMSE: {test_metrics['rmse_test']:.3f}")
        print(f"  - MAE: {test_metrics['mae_test']:.3f}")
        print(f"  - R²: {test_metrics['r2_test']:.3f}")
        print(f"  - MAPE: {test_metrics['mape_test']:.2f}%")
        
        return test_metrics, results_df
    
    def save_best_model(self, output_dir="models"):
        """En iyi modeli kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Model ve preprocessing nesnelerini kaydet
        model_artifacts = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': type(self.best_model).__name__
        }
        
        with open(f"{output_dir}/best_model.pkl", 'wb') as f:
            pickle.dump(model_artifacts, f)
        
        print(f"\n💾 En iyi model '{output_dir}/best_model.pkl' olarak kaydedildi")
    
    def run_training_pipeline(self):
        """Tüm training pipeline'ını çalıştır"""
        mlflow.set_experiment("delivery-time-prediction")
        
        with mlflow.start_run(run_name="full_training_pipeline"):
            # Veriyi yükle
            features, target = self.load_features()
            
            # Train/val/test split
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(features, target)
            
            # MLflow'a veri boyutlarını logla
            mlflow.log_params({
                "n_features": len(self.feature_names),
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "n_test_samples": len(X_test)
            })
            
            # Modelleri eğit
            self.train_linear_models(X_train, y_train, X_val, y_val)
            self.train_tree_models(X_train, y_train, X_val, y_val)
            
            # XGBoost kuruluysa eğit
            try:
                self.train_xgboost(X_train, y_train, X_val, y_val)
            except:
                print("⚠️ XGBoost bulunamadı, atlanıyor...")
            
            # En iyi model ile test
            test_metrics, results_df = self.final_evaluation(X_test, y_test)
            
            # MLflow'a test metriklerini logla
            mlflow.log_metrics(test_metrics)
            mlflow.log_param("best_model_type", type(self.best_model).__name__)
            
            # En iyi modeli kaydet
            self.save_best_model()
            mlflow.log_artifact("models/best_model.pkl")
            
            print(f"\n✅ Training pipeline tamamlandı!")
            print(f"🏆 En iyi model: {type(self.best_model).__name__}")
            
            return self.best_model, results_df

if __name__ == "__main__":
    # Önce veri üretim ve feature engineering çalıştırılmalı
    # python src/data_generation/generate_logistics_data.py
    # python src/feature_engineering/feature_pipeline.py
    
    # Model training
    trainer = ModelTrainer()
    best_model, results = trainer.run_training_pipeline()