import os
import json
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/BestClassifier/Production"

# Set tracking URI and preload model
mlflow.set_tracking_uri(TRACKING_URI)
try:
    MODEL = mlflow.pyfunc.load_model(MODEL_URI)
    LOADED_FROM = MODEL_URI
except Exception as e:
    MODEL = None
    LOADED_FROM = None

@app.get("/")
def index():
    msg = {
        "message": "Flask is ready",
        "tracking_uri": TRACKING_URI,
        "model_uri": MODEL_URI,
        "loaded_from": LOADED_FROM,
        "how_to_use": {
            "POST /predict": {
                "json": {"X": [[0.1, -0.2, 0.3, "..."]]} ,
                "or": {"features": {"f0": 0.1, "f1": -0.2, "f2": 0.3}}
            }
        }
    }
    return jsonify(msg)

@app.post("/predict")
def predict():
    global MODEL
    if MODEL is None:
        # try to load on demand
        try:
            MODEL = mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as e:
            return jsonify({"error": "Model not available. Train and register a Production model first."}), 400

    data = request.get_json(force=True)
    if "X" in data:
        X = pd.DataFrame(data["X"])
    elif "features" in data and isinstance(data["features"], dict):
        # single row
        X = pd.DataFrame([data["features"]])
    else:
        return jsonify({"error": "Payload must contain 'X' or 'features'."}), 400

    preds = MODEL.predict(X)
    # Many sklearn classifiers return ndarray with ints
    preds = [int(x) for x in preds]
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)