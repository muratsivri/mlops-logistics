from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.utils.task_group import TaskGroup
import sys
import os

sys.path.append('/opt/airflow/dags/mlops-project')

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['mlops-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'logistics_ml_pipeline',
    default_args=default_args,
    description='Lojistik öneri sistemi ML pipeline',
    schedule_interval='@daily',  # Her gün çalış
    catchup=False,
    tags=['ml', 'training', 'logistics']
)

def check_data_quality(**context):
    """Veri kalitesi kontrolü"""
    import pandas as pd
    import great_expectations as ge
    
    orders_df = pd.read_csv('/data/raw/orders.csv')
    
    ge_df = ge.from_pandas(orders_df)
    
    results = []
    
    for col in ['customer_id', 'order_value', 'delivery_distance_km']:
        result = ge_df.expect_column_values_to_not_be_null(col)
        results.append(result)
    
    results.append(ge_df.expect_column_values_to_be_between('order_value', 0, 10000))
    results.append(ge_df.expect_column_values_to_be_between('delivery_distance_km', 0, 500))
    
    failed_expectations = [r for r in results if not r['success']]
    
    if failed_expectations:
        raise ValueError(f"Veri kalitesi kontrolleri başarısız: {failed_expectations}")
    
    return True

def generate_fresh_data(**context):
    """Yeni veri üret"""
    from src.data_generation.generate_logistics_data import LogisticsDataGenerator
    
    generator = LogisticsDataGenerator(
        n_customers=1000,
        n_couriers=50,
        n_warehouses=10,
        n_days=7  # Son 7 günlük veri
    )
    
    generator.save_data(output_dir="/data/raw/fresh")
    
    context['task_instance'].xcom_push(key='data_generated', value=True)

def merge_data(**context):
    """Eski ve yeni veriyi birleştir"""
    import pandas as pd
    import shutil
    
    existing_orders = pd.read_csv('/data/raw/orders.csv')
    
    fresh_orders = pd.read_csv('/data/raw/fresh/orders.csv')
    
    all_orders = pd.concat([existing_orders, fresh_orders])
    all_orders['order_date'] = pd.to_datetime(all_orders['order_date'])
    
    cutoff_date = datetime.now() - timedelta(days=90)
    all_orders = all_orders[all_orders['order_date'] >= cutoff_date]
    
    all_orders.to_csv('/data/raw/orders_merged.csv', index=False)
    
    for file in ['customers.csv', 'couriers.csv', 'warehouses.csv']:
        shutil.copy(f'/data/raw/fresh/{file}', f'/data/raw/{file}')
    

def run_feature_engineering(**context):
    """Feature engineering çalıştır"""
    from src.feature_engineering.feature_pipeline import FeatureEngineer
    
    fe = FeatureEngineer()
    features, target = fe.run_pipeline(
        data_dir="/data/raw",
        output_dir="/data/features_new"
    )
    
    context['task_instance'].xcom_push(key='n_features', value=len(features.columns))
    context['task_instance'].xcom_push(key='n_samples', value=len(features))
    

def train_models(**context):
    """Model eğitimi"""
    from src.models.training_pipeline import ModelTrainer
    
    trainer = ModelTrainer()
    best_model, results = trainer.run_training_pipeline()
    
    test_mae = results['error'].abs().mean()
    context['task_instance'].xcom_push(key='test_mae', value=test_mae)
    context['task_instance'].xcom_push(key='best_model_type', value=type(best_model).__name__)
    

def evaluate_model(**context):
    """Model değerlendirme ve karşılaştırma"""
    import pickle
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    
    with open('/models/best_model.pkl', 'rb') as f:
        new_model_artifacts = pickle.load(f)
    
    try:
        with open('/models/production_model.pkl', 'rb') as f:
            prod_model_artifacts = pickle.load(f)
        
        test_features = pd.read_csv('/data/features/test_features.csv')
        test_target = pd.read_csv('/data/features/test_target.csv')['target']
        
        new_mae = mean_absolute_error(test_target, new_model_artifacts['model'].predict(test_features))
        prod_mae = mean_absolute_error(test_target, prod_model_artifacts['model'].predict(test_features))
        
        improvement = (prod_mae - new_mae) / prod_mae * 100
        
        context['task_instance'].xcom_push(key='improvement_percentage', value=improvement)
        
        if improvement > 5:  # %5'ten fazla iyileşme
        else:
    
    except FileNotFoundError:

    import shutil
    from src.monitoring.metrics_collector import ModelRegistry
    
    registry = ModelRegistry()
    
    test_mae = context['task_instance'].xcom_pull(key='test_mae')
    
    registry.register_model(
        model_path='/models/best_model.pkl',
        model_name='delivery_time_predictor',
        version=datetime.now().strftime('%Y%m%d_%H%M%S'),
        metrics={'test_mae': test_mae},
    )
    
    shutil.copy('/models/best_model.pkl', '/models/production_model.pkl')
    

def update_monitoring(**context):
    """Monitoring sistemini güncelle"""
    from src.monitoring.metrics_collector import MonitoringService
    
    monitor = MonitoringService()
    
    train_data = pd.read_csv('/data/features/features.csv').sample(1000)
    
    config = {
        'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'baseline_data_path': '/data/monitoring/baseline.csv',
        'alert_thresholds': {
            'mae': 5.0,
            'drift_score': 0.7
        }
    }
    
    train_data.to_csv('/data/monitoring/baseline.csv', index=False)
    
    with open('/data/monitoring/config.json', 'w') as f:
        json.dump(config, f)
    

with dag:
    data_quality_task = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        provide_context=True
    )
    
    with TaskGroup("data_generation") as data_gen_group:
        generate_data_task = PythonOperator(
            task_id='generate_fresh_data',
            python_callable=generate_fresh_data,
            provide_context=True
        )
        
        merge_data_task = PythonOperator(
            task_id='merge_data',
            python_callable=merge_data,
            provide_context=True
        )
        
        generate_data_task >> merge_data_task
    
    feature_eng_task = PythonOperator(
        task_id='run_feature_engineering',
        python_callable=run_feature_engineering,
        provide_context=True
    )
    
    train_task = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
        provide_context=True
    )
    
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True
    )
    
        provide_context=True,
        trigger_rule='none_failed_min_one_success'
    )
    
    )
    
    update_monitoring_task = PythonOperator(
        task_id='update_monitoring',
        python_callable=update_monitoring,
        provide_context=True,
        trigger_rule='none_failed_min_one_success'
    )
    
    success_email = EmailOperator(
        task_id='send_success_email',
        to=['mlops-team@company.com'],
        subject='ML Pipeline Başarıyla Tamamlandı',
        html_content='''
        <h3>ML Pipeline Başarıyla Tamamlandı</h3>
        <p>Tarih: {{ ds }}</p>
        <p>Model performansı ve detaylar için MLflow UI'ı kontrol edin.</p>
        ''',
        trigger_rule='none_failed_min_one_success'
    )
    
    data_quality_task >> data_gen_group >> feature_eng_task >> train_task >> evaluate_task
