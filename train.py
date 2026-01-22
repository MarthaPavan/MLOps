import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    # 1. Load cleaned dataset
    df = pd.read_csv('./data/cleaned_data.csv')

    # 2. Define Target and Features
    target = 'time_in_hospital'
    features = [
        'age', 'race', 'gender', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses', 'max_glu_serum', 
        'A1Cresult', 'diabetesMed', 'change'
    ]
    
    X = df[features]
    y = df[target]

    # 3. One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation & Accuracy Calculation
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Custom Accuracy: Predicted stay is within 1 day of actual stay
    within_1_day = np.abs(y_test - y_pred) <= 1
    accuracy_within_1_day = np.mean(within_1_day) * 100

    print(f"--- Model Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f} days")
    print(f"R2 Score (Variance explained): {r2:.4f}")
    print(f"Accuracy (Predictions within 1 day of actual): {accuracy_within_1_day:.2f}%")

    # 7. Save Artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    
    metrics = {
        "mae": float(mae),
        "r2_score": float(r2),
        "accuracy_within_1_day": float(accuracy_within_1_day)
    }
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()