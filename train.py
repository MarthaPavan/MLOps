import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import os
import mlflow

# 1. Get URI from environment variable (set by GitHub Actions)
# 2. If not found, fall back to your manual local string
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5005")

mlflow.set_tracking_uri(TRACKING_URI)
# Configure MLflow to use your local sqlite database
# mlflow.set_tracking_uri("http://127.0.0.1:5005")
mlflow.set_experiment("Diabetes_Hospital_Stay_Training")

def main():

    # PART 1: DATA PREPARATION
    
    # Load cleaned dataset
    df = pd.read_csv('./data/cleaned_data.csv')

    target = 'time_in_hospital'
    features = [
        'age', 'race', 'gender', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses', 'max_glu_serum', 
        'A1Cresult', 'diabetesMed', 'change'
    ]
    
    X = df[features]
    y = df[target]

    # One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)
    model_columns = list(X.columns)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # PART 2: MODEL TRAINING & MLFLOW TRACKING
    
    with mlflow.start_run(run_name="Linear_Regression_Training"):
        
        # 1. Log Training Parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("num_features", len(features))

        # 2. Train the Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 3. Calculate Performance Metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 4. Log Metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # 5. Save Artifacts locally for reference
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")
        joblib.dump(model_columns, "artifacts/columns.pkl")
        
        # 6. Log Artifacts and Model to MLflow Tracking Server
        mlflow.log_artifact("artifacts/columns.pkl")
        mlflow.sklearn.log_model(model, artifact_path="artifacts")        

        print(f"Training Complete. Metrics: R2: {r2:.4f}, MSE: {mse:.4f}")
        print("Model and metrics have been logged to MLflow.")

if __name__ == "__main__":
    main()
