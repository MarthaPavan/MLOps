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
    # Ensure this file exists in your directory
    df = pd.read_csv('./data/cleaned_data.csv')

    # 2. Define Target and Features (The order here must match your input list)
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
    
    # We save the column names to ensure the sample input matches the training format
    model_columns = list(X.columns)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation & Accuracy Calculation
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    within_1_day = np.abs(y_test - y_pred) <= 1
    accuracy_within_1_day = np.mean(within_1_day) * 100

    print(f"--- Model Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f} days")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy (within 1 day): {accuracy_within_1_day:.2f}%")

    # 7. Save Artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(model_columns, "artifacts/model_columns.pkl")

    # ---------------------------------------------------------
    # NEW: SAMPLE INPUT AS A LIST AND PREDICTION
    # ---------------------------------------------------------
    
    # This list represents: [age, race, gender, labs, procs, meds, out, emer, inpat, diags, glu, a1c, med, change]
    sample_list = [55, 'Caucasian', 'Female', 35, 1, 15, 0, 0, 0, 9, 'None', '>7', 'Yes', 'No']

    # 1. Convert list to DataFrame using the 'features' list as column names
    sample_df = pd.DataFrame([sample_list], columns=features)

    # 2. Apply dummy encoding to the sample
    sample_encoded = pd.get_dummies(sample_df)

    # 3. Align with model columns (fills missing dummy columns with 0)
    sample_encoded = sample_encoded.reindex(columns=model_columns, fill_value=0)

    # 4. Get Prediction
    prediction = model.predict(sample_encoded)

    print(f"\n--- Prediction for Sample Input ---")
    print(f"Input List: {sample_list}")
    print(f"Predicted target (time_in_hospital): {prediction[0]:.2f} days")

if __name__ == "__main__":
    main()