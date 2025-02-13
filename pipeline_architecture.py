import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

# Load Pre-trained Model
MODEL_PATH = "models/risk_model.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load("models/scaler.pkl")

# Initialize Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.get_json()
        df = pd.DataFrame(data)

        # Preprocess Data
        processed_data = scaler.transform(df)
        
        # Make Prediction
        predictions = model.predict_proba(processed_data)[:, 1]  # Get risk scores
        
        # Return Predictions
        response = {"risk_scores": predictions.tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
