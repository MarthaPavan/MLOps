import pandas as pd
import numpy as np
import os
import json
import joblib
import sys # 1. Import sys to handle CLI arguments
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    # ---------------------------------------------------------
    # PART 1: TRAINING (Setup for the model)
    # ---------------------------------------------------------
    
    # Load cleaned dataset - Ensure the path matches your environment
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

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save Artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(model_columns, "artifacts/columns.pkl")


if __name__ == "__main__":
    main()
