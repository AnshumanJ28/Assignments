"""
app.py
-------
Flask REST API for Heart Disease Prediction (Task 3).

Endpoints:
    GET  /            -> simple HTML form (optional UI) / health info
    GET  /health       -> health check, returns {"status": "ok"}
    POST /predict      -> accepts patient details as JSON, returns prediction as JSON

Run locally:
    python app.py

Example request:
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
               "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
               "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
             }'
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts once, at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_NAMES = joblib.load(FEATURES_PATH)


@app.route("/", methods=["GET"])
def home():
    """Basic landing page / health info."""
    try:
        return render_template("index.html", features=FEATURE_NAMES)
    except Exception:
        return jsonify({
            "message": "Heart Disease Prediction API is running.",
            "usage": "POST patient details as JSON to /predict",
            "required_fields": FEATURE_NAMES
        })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient details as JSON input and returns the prediction as JSON.

    Expected JSON body (all fields required):
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    """
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        # Validate that all required fields are present
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Build the feature vector in the correct order
        features = np.array([[data[f] for f in FEATURE_NAMES]], dtype=float)

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "probability_of_heart_disease": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local development only. Render/production uses gunicorn (see Procfile).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
