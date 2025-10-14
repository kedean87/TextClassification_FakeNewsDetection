# app.py
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Determine base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer using full paths
model_path = os.path.join(BASE_DIR, "best_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "best_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    # Convert numpy.int64 → Python int for JSON serialization
    return jsonify({"prediction": int(prediction)})

# No need for app.run() — Gunicorn will handle it in Azure

