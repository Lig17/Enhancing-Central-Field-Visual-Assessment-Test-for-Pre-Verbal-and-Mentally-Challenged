import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
from gaze_tracking_utils import GazeTracker
from model_inference import batch_inference

MODEL_PATHS = [
    'rf_model.pkl',
    'gb_model.pkl',
    'xgboost_model.pkl',
    'lightgbm_model.pkl',
    'stacking_model.pkl',
    'voting_model.pkl',
    'bagging_model.pkl'
]

def load_models():
    models = {}
    for path in MODEL_PATHS:
        try:
            models[path] = joblib.load(path)
        except:
            st.warning(f"Could not load {path}")
    return models

@st.cache_resource
def start_gaze_tracker():
    tracker = GazeTracker()
    return tracker

def run_gaze_tracking(tracker):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, ear, gaze_ratio = tracker.process_frame(frame)

        cv2.putText(processed_frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(processed_frame, f"Gaze Ratio: {gaze_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stframe.image(processed_frame, channels="BGR")

        if st.button("Stop Gaze Tracking"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.save_gaze_data()

def run_model_inference(tracker, models):
    st.subheader("Run Vision Assessment Models")
    if st.button("Analyze Gaze Data"):
        gaze_data = tracker.get_smoothed_gaze_data()
        if gaze_data:
            sample_input = np.mean(gaze_data)
            results = batch_inference(models.keys(), [sample_input])

            for model, result in results.items():
                st.write(f"**Model:** {model}")
                st.write(f"Prediction: {result['prediction']}")
                st.write(f"Probability: {result['probability']}\n")
        else:
            st.warning("No gaze data available.")

st.title("Central Field Visual Assessment")
st.sidebar.title("Navigation")

option = st.sidebar.radio("Go to:", ("Gaze Tracker", "Model Inference", "View Gaze Data"))

tracker = start_gaze_tracker()
models = load_models()

if option == "Gaze Tracker":
    st.header("Real-Time Gaze Tracking")
    if st.button("Start Gaze Tracking"):
        run_gaze_tracking(tracker)

elif option == "Model Inference":
    run_model_inference(tracker, models)

elif option == "View Gaze Data":
    st.subheader("Recorded Gaze Data")
    try:
        gaze_data = pd.read_csv('gaze_data.csv')
        st.line_chart(gaze_data['gaze_ratio'])
    except FileNotFoundError:
        st.warning("No gaze data found. Please run the gaze tracker first.")
