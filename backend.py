from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# 1. Load the Model and the Column Names saved during training
# These should be in your /artifacts folder
try:
    model = joblib.load('artifacts/model.pkl')
    model_columns = joblib.load('artifacts/model_columns.pkl')
except Exception as e:
    print(f"Error loading artifacts: {e}")

# The EXACT order of features as sent in your curl command
FEATURES_ORDER = [
    'age', 'race', 'gender', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'max_glu_serum',
    'A1Cresult', 'diabetesMed', 'change'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        # Create a DataFrame from the list
        # input_list will be: [55, "Caucasian", "Female", ...]
        input_list = data['features']
        input_df = pd.DataFrame([input_list], columns=FEATURES_ORDER)

        # 2. Transform the categorical text into Dummy Variables (0s and 1s)
        input_encoded = pd.get_dummies(input_df)

        # 3. Align with Training Columns
        # This is the "Magic" step: it adds missing columns like 'race_Asian' as 0
        # so the model doesn't see a string, it sees a numeric matrix.
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

        # 4. Predict
        prediction = model.predict(input_final)

        return jsonify({
            'status': 'success',
            'predicted_days': round(float(prediction[0]), 2)
        })
    except Exception as e:
        # This will help you see the EXACT error in the terminal if it fails again
       return jsonify({'error': str(e), 'type': str(type(e))}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5003, debug=True)