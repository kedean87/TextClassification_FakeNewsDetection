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

@app.route('/')
def home():
    return jsonify({"message": "Welcome to AG News Classifier API!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run listens on port 8080
    app.run(host="0.0.0.0", port=port)

