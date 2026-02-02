import os
import mlflow
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Resolve Tracking URI
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5005")
print(f"DEBUG: Connecting to MLflow at: {TRACKING_URI}") # DO NOT REMOVE THIS UNTIL GREEN

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Diabetes_Hospital_Stay_Training")

def main():
    # Load data
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
    X = pd.get_dummies(X, drop_first=True)
    model_columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Linear_Regression_Training"):
        mlflow.log_params({"test_size": 0.2, "random_state": 42, "model_type": "LinearRegression"})

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)

        # Artifact management
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")
        joblib.dump(model_columns, "artifacts/columns.pkl")
        
        mlflow.log_artifact("artifacts/columns.pkl")
        
        # Use 'name' instead of 'artifact_path' for newer MLflow versions
        # and register the model to bridge the CI/CD pipeline
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            registered_model_name="DiabetesLRModel"
        )        

        print(f"Training Complete. R2: {metrics['r2_score']:.4f}")

if __name__ == "__main__":
    main()
