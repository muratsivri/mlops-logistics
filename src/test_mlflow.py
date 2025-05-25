import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np


mlflow.set_tracking_uri("http://localhost:5000")


mlflow.set_experiment("logistics-recommendation-test")


X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run(run_name="test-run"):

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_size", 0.2)
    

    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)
    
 
    mlflow.sklearn.log_model(model, "model")
    
