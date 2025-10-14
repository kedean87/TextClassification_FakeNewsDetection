import json
import joblib

# Load pickled model/vectorizer (make sure these are included in the container)
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("best_vectorizer.pkl")

label_map = {0: "Fake", 1: "Real"}

def lambda_handler(event, context):
    """
    Lambda handler function for AG News prediction.
    Expects event to contain: {"text": "<text to classify>"}
    """
    try:
        text = event.get("text", "")
        if not text:
            return {"statusCode": 400, "body": json.dumps({"error": "No text provided"})}

        # Vectorize and predict
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": label_map[pred]})
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

