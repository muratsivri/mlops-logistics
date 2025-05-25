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
        
        features = pd.read_csv(f"{features_dir}/features.csv")
        target = pd.read_csv(f"{features_dir}/target.csv")['target']
        
        with open(f"{features_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{features_dir}/label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(f"{features_dir}/feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)
        
        
        return features, target
    
    def split_data(self, features, target, test_size=0.2, val_size=0.1, random_state=42):
        """Train, validation ve test setlerine ayır"""
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        
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
        
        with mlflow.start_run(run_name="linear_regression", nested=True):
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            train_metrics, _ = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(model, X_val, y_val, "val")
            
            mlflow.log_params({"model_type": "LinearRegression"})
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(model, "model")
            
            self.models['linear'] = model
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = model
        
        with mlflow.start_run(run_name="ridge_regression", nested=True):
            param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
            model = Ridge()
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            mlflow.log_params({
                "model_type": "Ridge",
                "best_alpha": grid_search.best_params_['alpha']
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(best_model, "model")
            
            self.models['ridge'] = best_model
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
    
    def train_tree_models(self, X_train, y_train, X_val, y_val):
        """Tree-based modelleri eğit"""
        
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
            
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            mlflow.log_params({
                "model_type": "RandomForest",
                **grid_search.best_params_
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
            
            self.models['random_forest'] = best_model
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
        
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
            
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            mlflow.log_params({
                "model_type": "GradientBoosting",
                **grid_search.best_params_
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.sklearn.log_model(best_model, "model")
            
            self.models['gradient_boosting'] = best_model
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost modelini eğit"""
        
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
            
            train_metrics, _ = self.evaluate_model(best_model, X_train, y_train, "train")
            val_metrics, _ = self.evaluate_model(best_model, X_val, y_val, "val")
            
            mlflow.log_params({
                "model_type": "XGBoost",
                **grid_search.best_params_
            })
            mlflow.log_metrics({**train_metrics, **val_metrics})
            mlflow.xgboost.log_model(best_model, "model")
            
            self.models['xgboost'] = best_model
            
            if val_metrics['rmse_val'] < self.best_score:
                self.best_score = val_metrics['rmse_val']
                self.best_model = best_model
    
    def final_evaluation(self, X_test, y_test):
        """En iyi model ile final değerlendirme"""
        
        test_metrics, predictions = self.evaluate_model(
            self.best_model, X_test, y_test, "test"
        )
        
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions,
            'error': y_test - predictions,
            'error_percentage': np.abs((y_test - predictions) / y_test) * 100
        })
        
        
        return test_metrics, results_df
    
    def save_best_model(self, output_dir="models"):
        """En iyi modeli kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_artifacts = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': type(self.best_model).__name__
        }
        
        with open(f"{output_dir}/best_model.pkl", 'wb') as f:
            pickle.dump(model_artifacts, f)
        
    
    def run_training_pipeline(self):
        """Tüm training pipeline'ını çalıştır"""
        mlflow.set_experiment("delivery-time-prediction")
        
        with mlflow.start_run(run_name="full_training_pipeline"):
            features, target = self.load_features()
            
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(features, target)
            
            mlflow.log_params({
                "n_features": len(self.feature_names),
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "n_test_samples": len(X_test)
            })
            
            self.train_linear_models(X_train, y_train, X_val, y_val)
            self.train_tree_models(X_train, y_train, X_val, y_val)
            
            try:
                self.train_xgboost(X_train, y_train, X_val, y_val)
            except:
            
            test_metrics, results_df = self.final_evaluation(X_test, y_test)
            
            mlflow.log_metrics(test_metrics)
            mlflow.log_param("best_model_type", type(self.best_model).__name__)
            
            self.save_best_model()
            mlflow.log_artifact("models/best_model.pkl")
            
            
            return self.best_model, results_df

if __name__ == "__main__":
    
    trainer = ModelTrainer()
    best_model, results = trainer.run_training_pipeline()