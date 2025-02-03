import joblib
import numpy as np
import pandas as pd

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def preprocess_input(input_data):
    if isinstance(input_data, list):
        input_data = np.array(input_data).reshape(1, -1)
    elif isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    return input_data

def make_prediction(model, input_data):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None
    return prediction, probability

def batch_inference(models, input_data):
    results = {}
    for model_path in models:
        model = load_model(model_path)
        prediction, probability = make_prediction(model, input_data)
        results[model_path] = {
            'prediction': prediction.tolist(),
            'probability': probability.tolist() if probability is not None else 'N/A'
        }
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
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']}\n")
