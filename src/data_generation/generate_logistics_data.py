import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

class LogisticsDataGenerator:
    """Lojistik öneri sistemi için sentetik veri üretici"""
    
    def __init__(self, n_customers=1000, n_couriers=50, n_warehouses=10, n_days=90):
        self.n_customers = n_customers
        self.n_couriers = n_couriers
        self.n_warehouses = n_warehouses
        self.n_days = n_days
        
        self.cities = {
            "Istanbul": ["Kadıköy", "Üsküdar", "Beşiktaş", "Şişli", "Bakırköy", "Maltepe", "Kartal"],
            "Ankara": ["Çankaya", "Keçiören", "Mamak", "Yenimahalle", "Etimesgut"],
            "İzmir": ["Karşıyaka", "Bornova", "Konak", "Buca", "Bayraklı"],
            "Bursa": ["Nilüfer", "Osmangazi", "Yıldırım"],
            "Antalya": ["Muratpaşa", "Kepez", "Konyaaltı"]
        }
        
        self.product_categories = ["Elektronik", "Giyim", "Gıda", "Kitap", "Kozmetik", "Ev", "Spor", "Oyuncak"]
        
    def generate_customers(self):
        """Müşteri verisi üret"""
        customers = []
        
        for i in range(self.n_customers):
            city = random.choice(list(self.cities.keys()))
            district = random.choice(self.cities[city])
            
            customer = {
                "customer_id": f"CUST_{i:05d}",
                "age": np.random.randint(18, 70),
                "city": city,
                "district": district,
                "registration_date": datetime.now() - timedelta(days=np.random.randint(0, 365*2)),
                "preferred_delivery_time": random.choice(["morning", "afternoon", "evening"]),
                "premium_member": random.choices([True, False], weights=[0.3, 0.7])[0],
                "avg_order_value": np.random.gamma(2, 50),  # Gamma dağılımı ile daha gerçekçi
                "total_orders": np.random.poisson(10),
                "email": f"customer_{i}@example.com",
                "phone": f"+905{np.random.randint(100000000, 999999999)}"
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_orders(self, customers_df):
        """Sipariş verisi üret"""
        orders = []
        order_id = 0
        
        start_date = datetime.now() - timedelta(days=self.n_days)
        
        for day in range(self.n_days):
            current_date = start_date + timedelta(days=day)
            
            if current_date.weekday() in [5, 6]:  # Hafta sonu
                daily_orders = np.random.poisson(self.n_customers * 0.6)
            elif current_date.day in [11, 25]:  # İndirim günleri (11.11, ayın 25'i)
                daily_orders = np.random.poisson(self.n_customers * 1.2)
            else:
                daily_orders = np.random.poisson(self.n_customers * 0.4)
            
            for _ in range(daily_orders):
                customer = customers_df.sample(1).iloc[0]
                
                if customer["preferred_delivery_time"] == "morning":
                    hour = np.random.choice(range(6, 12), p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
                elif customer["preferred_delivery_time"] == "afternoon":
                    hour = np.random.choice(range(12, 18), p=[0.15, 0.2, 0.25, 0.2, 0.15, 0.05])
                else:  # evening
                    hour = np.random.choice(range(18, 23), p=[0.3, 0.25, 0.2, 0.15, 0.1])
                
                order_datetime = current_date.replace(hour=hour, minute=np.random.randint(0, 60))
                
                order = {
                    "order_id": f"ORD_{order_id:07d}",
                    "customer_id": customer["customer_id"],
                    "order_datetime": order_datetime,
                    "order_date": order_datetime.date(),
                    "order_hour": hour,
                    "product_category": random.choice(self.product_categories),
                    "order_value": np.random.gamma(2, customer["avg_order_value"]/2),
                    "items_count": np.random.poisson(3) + 1,
                    "weight_kg": np.random.gamma(2, 2),
                    "volume_cm3": np.random.gamma(3, 1000),
                    "fragile": random.choices([True, False], weights=[0.2, 0.8])[0],
                    "perishable": random.choices([True, False], weights=[0.1, 0.9])[0],
                    "warehouse_id": f"WH_{np.random.randint(self.n_warehouses):02d}",
                    "courier_id": f"COURIER_{np.random.randint(self.n_couriers):03d}",
                    "delivery_distance_km": np.random.gamma(2, 5),
                    "traffic_condition": random.choice(["low", "medium", "high"]),
                    "weather": random.choice(["sunny", "rainy", "cloudy", "snowy"]),
                    "promised_delivery_hours": random.choice([24, 48, 72]),
                    "actual_delivery_hours": None,
                    "delivery_rating": None,
                    "delivery_attempt_count": 1,
                    "payment_method": random.choice(["credit_card", "debit_card", "cash", "digital_wallet"]),
                    "order_status": "delivered"
                }
                
                base_delivery_time = order["delivery_distance_km"] * 1.5  # Base: 1.5 saat/km
                
                factors = {
                    "traffic": {"low": 1.0, "medium": 1.3, "high": 1.8},
                    "weather": {"sunny": 1.0, "cloudy": 1.1, "rainy": 1.4, "snowy": 2.0},
                    "fragile": 1.2 if order["fragile"] else 1.0,
                    "perishable": 0.8 if order["perishable"] else 1.0,  # Hızlı teslimat
                    "premium": 0.7 if customer["premium_member"] else 1.0,
                    "time_of_day": 1.5 if hour in [17, 18, 19] else 1.0  # Akşam trafiği
                }
                
                total_factor = (factors["traffic"][order["traffic_condition"]] * 
                               factors["weather"][order["weather"]] * 
                               factors["fragile"] * 
                               factors["perishable"] * 
                               factors["premium"] * 
                               factors["time_of_day"])
                
                order["actual_delivery_hours"] = base_delivery_time * total_factor + np.random.normal(0, 2)
                order["actual_delivery_hours"] = max(0.5, order["actual_delivery_hours"])  # Min 30 dakika
                
                delay_ratio = order["actual_delivery_hours"] / order["promised_delivery_hours"]
                
                if delay_ratio <= 0.5:  # Çok erken teslimat
                    order["delivery_rating"] = 5
                elif delay_ratio <= 1.0:  # Zamanında
                    order["delivery_rating"] = np.random.choice([4, 5], p=[0.3, 0.7])
                elif delay_ratio <= 1.2:  # Biraz geç
                    order["delivery_rating"] = np.random.choice([3, 4], p=[0.6, 0.4])
                else:  # Çok geç
                    order["delivery_rating"] = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                    order["delivery_attempt_count"] = np.random.choice([2, 3], p=[0.7, 0.3])
                
                orders.append(order)
                order_id += 1
        
        return pd.DataFrame(orders)
    
    def generate_couriers(self):
        """Kurye verisi üret"""
        couriers = []
        
        vehicle_types = {
            "motorcycle": {"speed": 1.2, "capacity": 50},
            "car": {"speed": 1.0, "capacity": 200},
            "bicycle": {"speed": 2.0, "capacity": 20},
            "van": {"speed": 0.8, "capacity": 500},
            "scooter": {"speed": 1.5, "capacity": 30}
        }
        
        for i in range(self.n_couriers):
            vehicle = random.choice(list(vehicle_types.keys()))
            
            courier = {
                "courier_id": f"COURIER_{i:03d}",
                "name": f"Kurye {i}",
                "experience_years": np.random.gamma(2, 2),
                "vehicle_type": vehicle,
                "vehicle_capacity_kg": vehicle_types[vehicle]["capacity"],
                "avg_speed_factor": vehicle_types[vehicle]["speed"],
                "rating": min(5.0, 3.0 + np.random.exponential(0.5)),  # 3-5 arası, 5'e yakın yoğun
                "total_deliveries": np.random.poisson(500),
                "on_time_delivery_rate": np.random.beta(8, 2),  # Çoğu kurye iyi performans
                "available_hours_per_day": np.random.choice([4, 6, 8, 10], p=[0.1, 0.2, 0.5, 0.2]),
                "preferred_districts": random.sample(
                    [d for districts in self.cities.values() for d in districts], 
                    k=random.randint(3, 7)
                )
            }
            couriers.append(courier)
        
        return pd.DataFrame(couriers)
    
    def generate_warehouses(self):
        """Depo verisi üret"""
        warehouses = []
        
        for i in range(self.n_warehouses):
            city = random.choice(list(self.cities.keys()))
            
            warehouse = {
                "warehouse_id": f"WH_{i:02d}",
                "name": f"{city} Depo {i}",
                "city": city,
                "district": random.choice(self.cities[city]),
                "capacity_m3": np.random.choice([1000, 2000, 5000, 10000]),
                "current_load_percentage": np.random.uniform(30, 90),
                "staff_count": np.random.poisson(20),
                "loading_docks": np.random.choice([2, 4, 6, 8]),
                "avg_processing_time_minutes": np.random.gamma(2, 15),
                "operating_hours_start": 6,
                "operating_hours_end": 22,
                "has_cold_storage": random.choice([True, False]),
                "has_fragile_handling": random.choice([True, False]),
                "automation_level": random.choice(["low", "medium", "high"])
            }
            warehouses.append(warehouse)
        
        return pd.DataFrame(warehouses)
    
    def generate_delivery_zones(self):
        """Teslimat bölgesi verisi üret"""
        zones = []
        zone_id = 0
        
        for city, districts in self.cities.items():
            for district in districts:
                zone = {
                    "zone_id": f"ZONE_{zone_id:03d}",
                    "city": city,
                    "district": district,
                    "avg_traffic_morning": np.random.uniform(1.0, 2.5),
                    "avg_traffic_afternoon": np.random.uniform(1.2, 3.0),
                    "avg_traffic_evening": np.random.uniform(1.5, 3.5),
                    "population_density": np.random.gamma(2, 1000),
                    "business_density": np.random.gamma(1.5, 500),
                    "residential_density": np.random.gamma(2, 800),
                    "avg_building_floors": np.random.choice([2, 4, 6, 10, 15]),
                    "parking_difficulty": random.choice(["low", "medium", "high"]),
                    "delivery_point_types": {
                        "home": np.random.uniform(0.4, 0.7),
                        "office": np.random.uniform(0.2, 0.4),
                        "shop": np.random.uniform(0.1, 0.2)
                    }
                }
                zones.append(zone)
                zone_id += 1
        
        return pd.DataFrame(zones)
    
    def save_data(self, output_dir="data/raw"):
        """Tüm veriyi kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        
        customers_df = self.generate_customers()
        customers_df.to_csv(f"{output_dir}/customers.csv", index=False)
        
        orders_df = self.generate_orders(customers_df)
        orders_df.to_csv(f"{output_dir}/orders.csv", index=False)
        
        couriers_df = self.generate_couriers()
        couriers_df.to_csv(f"{output_dir}/couriers.csv", index=False)
        
        warehouses_df = self.generate_warehouses()
        warehouses_df.to_csv(f"{output_dir}/warehouses.csv", index=False)
        
        zones_df = self.generate_delivery_zones()
        zones_df.to_csv(f"{output_dir}/delivery_zones.csv", index=False)
        
        summary_stats = {
            "total_orders": len(orders_df),
            "total_customers": len(customers_df),
            "total_couriers": len(couriers_df),
            "total_warehouses": len(warehouses_df),
            "total_zones": len(zones_df),
            "avg_order_value": float(orders_df["order_value"].mean()),
            "avg_delivery_time": float(orders_df["actual_delivery_hours"].mean()),
            "avg_delivery_rating": float(orders_df["delivery_rating"].mean()),
            "date_range": f"{orders_df['order_date'].min()} - {orders_df['order_date'].max()}"
        }
        
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "parameters": {
                "n_customers": self.n_customers,
                "n_couriers": self.n_couriers,
                "n_warehouses": self.n_warehouses,
                "n_days": self.n_days
            },
            "statistics": summary_stats
        }
        
        with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        
        
        return customers_df, orders_df, couriers_df, warehouses_df, zones_df

if __name__ == "__main__":
    generator = LogisticsDataGenerator(
        n_customers=1000,
        n_couriers=50,
        n_warehouses=10,
        n_days=90
    )
    
    generator.save_data()