import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
import pickle
import os
from datetime import datetime
import mlflow
import mlflow.sklearn

class FeatureEngineer:
    """Lojistik öneri sistemi için feature engineering"""
    
    def __init__(self, mlflow_tracking_uri="http://localhost:5000"):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
    def load_data(self, data_dir="data/raw"):
        """Tüm veriyi yükle"""
        print("📂 Veri yükleniyor...")
        
        customers = pd.read_csv(f"{data_dir}/customers.csv")
        orders = pd.read_csv(f"{data_dir}/orders.csv")
        couriers = pd.read_csv(f"{data_dir}/couriers.csv")
        warehouses = pd.read_csv(f"{data_dir}/warehouses.csv")
        zones = pd.read_csv(f"{data_dir}/delivery_zones.csv")
        
        # Tarih sütunlarını datetime'a çevir
        orders['order_datetime'] = pd.to_datetime(orders['order_datetime'])
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        customers['registration_date'] = pd.to_datetime(customers['registration_date'])
        
        print(f"✅ Veri yüklendi: {len(orders)} sipariş")
        
        return customers, orders, couriers, warehouses, zones
    
    def create_customer_features(self, customers, orders):
        print("👥 Müşteri özellikleri oluşturuluyor...")

        for col in ['avg_order_value', 'order_count', 'avg_rating', 'std_rating',
                    'days_since_registration', 'premium_member', 'total_orders']:
            if col in customers.columns:
                customers = customers.drop(columns=[col])

        customer_stats = orders.groupby('customer_id').agg({
            'order_id': 'count',
            'order_value': ['mean', 'sum', 'std'],
            'actual_delivery_hours': ['mean', 'std'],
            'delivery_rating': ['mean', 'std'],
            'items_count': 'mean',
            'delivery_attempt_count': 'mean'
        }).reset_index()

        customer_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in customer_stats.columns]

        customer_stats = customer_stats.rename(columns={
            'customer_id_': 'customer_id',
            'order_id_count': 'order_count',
            'order_value_mean': 'avg_order_value',
            'order_value_sum': 'total_order_value',
            'order_value_std': 'std_order_value',
            'actual_delivery_hours_mean': 'avg_delivery_time',
            'actual_delivery_hours_std': 'std_delivery_time',
            'delivery_rating_mean': 'avg_rating',
            'delivery_rating_std': 'std_rating',
            'items_count_mean': 'avg_items_per_order',
            'delivery_attempt_count_mean': 'avg_delivery_attempts'
        })

        customer_features = customers.merge(customer_stats, on='customer_id', how='left')
        customer_features['days_since_registration'] = (
            datetime.now() - customer_features['registration_date']
        ).dt.days

        category_preferences = orders.groupby(['customer_id', 'product_category']).size().unstack(fill_value=0)
        category_preferences = category_preferences.div(category_preferences.sum(axis=1), axis=0)
        category_preferences = category_preferences.add_prefix('pref_category_')

        customer_features = customer_features.merge(
            category_preferences,
            left_on='customer_id',
            right_index=True,
            how='left'
        )

        orders['order_hour_bin'] = pd.cut(
            orders['order_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )

        time_preferences = orders.groupby(['customer_id', 'order_hour_bin']).size().unstack(fill_value=0)
        time_preferences = time_preferences.div(time_preferences.sum(axis=1), axis=0)
        time_preferences = time_preferences.add_prefix('pref_time_')

        customer_features = customer_features.merge(
            time_preferences,
            left_on='customer_id',
            right_index=True,
            how='left'
        )

        # Eğer premium_member sütunu yoksa, sahte olarak ekle
        if 'premium_member' not in customer_features.columns:
            customer_features['premium_member'] = False

        print("🧪 Final customer_features columns:", customer_features.columns.tolist())
        return customer_features

    
    def create_courier_features(self, couriers, orders):
        """Kurye bazlı özellikler"""
        print("🚚 Kurye özellikleri oluşturuluyor...")
        
        # Kurye performans metrikleri
        courier_performance = orders.groupby('courier_id').agg({
            'order_id': 'count',
            'actual_delivery_hours': 'mean',
            'delivery_rating': 'mean',
            'delivery_attempt_count': lambda x: (x > 1).sum() / len(x)  # Başarısız teslimat oranı
        }).reset_index()
        
        courier_performance.columns = ['courier_id', 'total_deliveries_recent', 
                                      'avg_delivery_time_recent', 'avg_rating_recent',
                                      'failed_delivery_rate']
        
        # Kurye bilgileriyle birleştir
        courier_features = couriers.merge(courier_performance, on='courier_id', how='left')
        
        # Deneyim kategorisi
        courier_features['experience_category'] = pd.cut(
            courier_features['experience_years'],
            bins=[0, 1, 3, 5, 100],
            labels=['junior', 'mid', 'senior', 'expert']
        )
        
        # Araç kapasitesi kategorisi
        courier_features['capacity_category'] = pd.cut(
            courier_features['vehicle_capacity_kg'],
            bins=[0, 50, 200, 1000],
            labels=['small', 'medium', 'large']
        )
        
        return courier_features
    
    def create_warehouse_features(self, warehouses, orders):
        """Depo bazlı özellikler"""
        print("🏭 Depo özellikleri oluşturuluyor...")
        
        # Depo yoğunluk metrikleri
        warehouse_load = orders.groupby('warehouse_id').agg({
            'order_id': 'count',
            'weight_kg': 'sum',
            'volume_cm3': 'sum',
            'actual_delivery_hours': 'mean'
        }).reset_index()
        
        warehouse_load.columns = ['warehouse_id', 'order_count', 'total_weight',
                                 'total_volume', 'avg_delivery_time_from_warehouse']
        
        # Depo bilgileriyle birleştir
        warehouse_features = warehouses.merge(warehouse_load, on='warehouse_id', how='left')
        
        # Kapasite kullanım oranı
        warehouse_features['capacity_utilization'] = (
            warehouse_features['current_load_percentage'] / 100
        )
        
        # Verimlilik skoru
        warehouse_features['efficiency_score'] = (
            warehouse_features['order_count'] / 
            (warehouse_features['staff_count'] * warehouse_features['loading_docks'])
        )
        
        return warehouse_features
    
    def create_order_features(self, orders, customer_features, courier_features, 
                            warehouse_features, zones):
        """Sipariş bazlı özellikler"""
        print("📦 Sipariş özellikleri oluşturuluyor...")
        
        # Temel birleştirmeler
        order_features = orders.merge(
            customer_features[['customer_id', 'premium_member', 'avg_order_value', 
                             'order_count', 'avg_rating', 'days_since_registration']],
            on='customer_id', how='left'
        )
        
        order_features = order_features.merge(
            courier_features[['courier_id', 'vehicle_type', 'experience_years', 
                            'rating', 'on_time_delivery_rate']],
            on='courier_id', how='left'
        )
        
        order_features = order_features.merge(
            warehouse_features[['warehouse_id', 'automation_level', 'capacity_utilization',
                              'avg_processing_time_minutes']],
            on='warehouse_id', how='left'
        )
        
        # Zaman özellikleri
        order_features['order_hour'] = order_features['order_datetime'].dt.hour
        order_features['order_day_of_week'] = order_features['order_datetime'].dt.dayofweek
        order_features['is_weekend'] = order_features['order_day_of_week'].isin([5, 6]).astype(int)
        order_features['is_peak_hour'] = order_features['order_hour'].isin([11, 12, 17, 18, 19]).astype(int)
        
        # Sipariş özellikleri
        order_features['value_per_item'] = order_features['order_value'] / order_features['items_count']
        order_features['weight_per_item'] = order_features['weight_kg'] / order_features['items_count']
        order_features['volume_per_item'] = order_features['volume_cm3'] / order_features['items_count']
        
        # Karmaşıklık skoru
        order_features['complexity_score'] = (
            order_features['items_count'] * 0.3 +
            order_features['weight_kg'] * 0.2 +
            (order_features['fragile'].astype(int) * 5) +
            (order_features['perishable'].astype(int) * 10) +
            order_features['delivery_distance_km'] * 0.5
        )
        
        # Trafik ve hava durumu etki skoru
        traffic_impact = {'low': 1, 'medium': 1.5, 'high': 2.5}
        weather_impact = {'sunny': 1, 'cloudy': 1.2, 'rainy': 1.8, 'snowy': 2.5}
        
        order_features['traffic_impact'] = order_features['traffic_condition'].map(traffic_impact)
        order_features['weather_impact'] = order_features['weather'].map(weather_impact)
        
        # Teslimat zorluğu skoru
        order_features['delivery_difficulty_score'] = (
            order_features['complexity_score'] * 
            order_features['traffic_impact'] * 
            order_features['weather_impact']
        )
        
        return order_features
    
    def prepare_features_for_modeling(self, order_features, target_col='actual_delivery_hours'):
        """Model için özellikleri hazırla"""
        print("🔧 Model özellikleri hazırlanıyor...")
        
        # Hedef değişkeni ayır
        if target_col in order_features.columns:
            y = order_features[target_col]
            order_features = order_features.drop(columns=[target_col])
        else:
            y = None
        
        # Kategorik değişkenler
        categorical_columns = [
            'preferred_delivery_time', 'product_category', 'traffic_condition',
            'weather', 'payment_method', 'vehicle_type', 'automation_level',
            'city_x', 'district_x'
        ]
        
        # Numerik değişkenler
        numeric_columns = [
            'age', 'avg_order_value', 'order_count', 'avg_rating',
            'days_since_registration', 'order_value', 'items_count',
            'weight_kg', 'volume_cm3', 'delivery_distance_km',
            'experience_years', 'rating', 'on_time_delivery_rate',
            'capacity_utilization', 'avg_processing_time_minutes',
            'order_hour', 'order_day_of_week', 'is_weekend', 'is_peak_hour',
            'value_per_item', 'weight_per_item', 'volume_per_item',
            'complexity_score', 'traffic_impact', 'weather_impact',
            'delivery_difficulty_score'
        ]
        
        # Boolean değişkenler
        boolean_columns = ['premium_member', 'fragile', 'perishable']
        
        # Mevcut sütunları kontrol et
        categorical_columns = [col for col in categorical_columns if col in order_features.columns]
        numeric_columns = [col for col in numeric_columns if col in order_features.columns]
        boolean_columns = [col for col in boolean_columns if col in order_features.columns]
        
        # One-hot encoding
        print("  - Kategorik değişkenler encode ediliyor...")
        encoded_features = []
        
        for col in categorical_columns:
            # Label encoding
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                order_features[col] = self.label_encoders[col].fit_transform(order_features[col].fillna('missing'))
            else:
                order_features[col] = self.label_encoders[col].transform(order_features[col].fillna('missing'))
            
            # One-hot encoding
            one_hot = pd.get_dummies(order_features[col], prefix=col)
            encoded_features.append(one_hot)
        
        # Numerik özellikleri normalize et
        print("  - Numerik değişkenler normalize ediliyor...")
        numeric_data = order_features[numeric_columns].fillna(0)
        
        if hasattr(self.scaler, 'mean_'):
            numeric_scaled = self.scaler.transform(numeric_data)
        else:
            numeric_scaled = self.scaler.fit_transform(numeric_data)
        
        numeric_df = pd.DataFrame(
            numeric_scaled, 
            columns=[f'scaled_{col}' for col in numeric_columns],
            index=order_features.index
        )
        
        # Boolean değişkenleri ekle
        boolean_data = order_features[boolean_columns].astype(int)
        
        # Tüm özellikleri birleştir
        final_features = pd.concat(
            [numeric_df, boolean_data] + encoded_features, 
            axis=1
        )
        
        # Özellik isimlerini kaydet
        self.feature_names = list(final_features.columns)
        
        print(f"✅ Toplam {len(self.feature_names)} özellik oluşturuldu")
        
        return final_features, y
    
    def save_features(self, features, y, output_dir="data/features"):
        """Özellikleri kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Features ve target'ı kaydet
        features.to_csv(f"{output_dir}/features.csv", index=False)
        if y is not None:
            y.to_csv(f"{output_dir}/target.csv", index=False, header=['target'])
        
        # Preprocessing nesnelerini kaydet
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f"{output_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(f"{output_dir}/feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"💾 Özellikler '{output_dir}' klasörüne kaydedildi")
    
    def run_pipeline(self, data_dir="data/raw", output_dir="data/features"):
        """Tüm feature engineering pipeline'ını çalıştır"""
        mlflow.set_experiment("feature-engineering")
        
        with mlflow.start_run(run_name="feature-pipeline"):
            # Veriyi yükle
            customers, orders, couriers, warehouses, zones = self.load_data(data_dir)
            
            # MLflow'a veri boyutlarını logla
            mlflow.log_param("n_customers", len(customers))
            mlflow.log_param("n_orders", len(orders))
            mlflow.log_param("n_couriers", len(couriers))
            mlflow.log_param("n_warehouses", len(warehouses))
            
            # Feature'ları oluştur
            customer_features = self.create_customer_features(customers, orders)
            courier_features = self.create_courier_features(couriers, orders)
            warehouse_features = self.create_warehouse_features(warehouses, orders)
            order_features = self.create_order_features(
                orders, customer_features, courier_features, warehouse_features, zones
            )
            
            # Model için hazırla
            features, y = self.prepare_features_for_modeling(order_features)
            
            # MLflow'a feature sayısını logla
            mlflow.log_metric("n_features", len(self.feature_names))
            mlflow.log_metric("n_samples", len(features))
            
            # Özellikleri kaydet
            self.save_features(features, y, output_dir)
            
            # MLflow'a artifact olarak kaydet
            mlflow.log_artifact(output_dir)
            
            print(f"\n✅ Feature engineering pipeline tamamlandı!")
            print(f"   - {len(features)} örnek")
            print(f"   - {len(self.feature_names)} özellik")
            
            return features, y

if __name__ == "__main__":
    # Feature engineering pipeline'ını çalıştır
    fe = FeatureEngineer()
    features, target = fe.run_pipeline()