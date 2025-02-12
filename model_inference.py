import joblib
import numpy as np
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cache for loaded models to avoid redundant loading
model_cache = {}

def load_model(model_path):
    """Load and cache a machine learning model."""
    if model_path in model_cache:
        return model_cache[model_path]

    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        model_cache[model_path] = model
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        return None

def preprocess_input(input_data):
    """Convert input data to the required format."""
    if isinstance(input_data, list):
        input_data = np.array(input_data).reshape(1, -1)
    elif isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
    else:
        logging.error("Invalid input data format. Must be list, NumPy array, or DataFrame.")
        raise ValueError("Input data must be a list, NumPy array, or Pandas DataFrame.")

    return input_data

def make_prediction(model, input_data):
    """Make a prediction using the given model."""
    try:
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None

        return {
            'prediction': prediction.tolist(),
            'probability': probability.tolist() if probability is not None else "N/A"
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"prediction": None, "probability": None}

def batch_inference(models, input_data):
    """Run inference across multiple models."""
    results = {}

    for model_path in models:
        model = load_model(model_path)
        if model is not None:
            results[model_path] = make_prediction(model, input_data)
        else:
            results[model_path] = {"error": "Model could not be loaded"}

    return results

if __name__ == "__main__":
    sample_input = [0.5, 1.2, -0.3, 0.8, 1.1, 0.4, -0.2, 0.9, 0.7, -0.1, 0.3, 1.5, 0.6, 0.2, -0.4]

    models = [
        'rf_model.pkl',
        'gb_model.pkl',
        'xgboost_model.pkl',
        'lightgbm_model.pkl',
        'stacking_model.pkl',
        'voting_model.pkl',
        'bagging_model.pkl'
    ]

    inference_results = batch_inference(models, sample_input)

    for model, result in inference_results.items():
        print(f"Model: {model}")
        print(f"Prediction: {result.get('prediction', 'N/A')}")
        print(f"Probability: {result.get('probability', 'N/A')}\n")
