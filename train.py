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

    # ---------------------------------------------------------
    # PART 2: DYNAMIC INPUT PARSING
    # ---------------------------------------------------------
    
    # Join all CLI arguments starting from index 1 to capture everything inside the brackets
    # This handles the case where the shell might split arguments by space
    input_args = " ".join(sys.argv[1:])
    
    if not input_args:
        print("Usage: python script.py [age race gender labs procs meds outpat emer inpat diags glu a1c med change]")
        return

    # Strip the brackets '[' and ']' and remove extra quotes
    clean_input = input_args.strip("[]").replace('"', '').replace("'", "")
    
    # Split by space to get individual values
    val_list = clean_input.split()

    if len(val_list) != 14:
        print(f"Error: Expected 14 values, but received {len(val_list)}.")
        return

    # Convert numeric fields (indices: 0, 3, 4, 5, 6, 7, 8, 9)
    # The rest remain as strings
    processed_list = []
    numeric_indices = [0, 3, 4, 5, 6, 7, 8, 9]
    
    for i, val in enumerate(val_list):
        if i in numeric_indices:
            processed_list.append(float(val))
        else:
            processed_list.append(val)

    # ---------------------------------------------------------
    # PART 3: PREDICTION
    # ---------------------------------------------------------
    
    # 1. Convert list to DataFrame
    sample_df = pd.DataFrame([processed_list], columns=features)

    # 2. Apply dummy encoding
    sample_encoded = pd.get_dummies(sample_df)

    # 3. Align with model columns (fills missing dummy columns with 0)
    sample_encoded = sample_encoded.reindex(columns=model_columns, fill_value=0)

    # 4. Get Prediction
    prediction = model.predict(sample_encoded)

    print(f"\n--- Prediction Results ---")
    print(f"Input Received: {processed_list}")
    print(f"Predicted days in hospital: {prediction[0]:.2f} days")

if __name__ == "__main__":
    main()