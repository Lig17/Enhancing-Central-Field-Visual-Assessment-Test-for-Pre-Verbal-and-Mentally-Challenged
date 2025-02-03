# Enhancing Central Field Visual Assessment Test

## 🚀 Project Overview

This project focuses on enhancing the precision of the **Central Field Visual Assessment Test** using advanced **Machine Learning**, **Eye-Gaze Tracking**, **Signal Processing**, and **Real-Time Data Analysis**. It aims to automate vision excellence recognition for **pre-verbal** and **mentally challenged individuals**.

## 🎯 Key Features

- **Real-Time Gaze Tracking**: Eye movement tracking using facial landmark detection.
- **ML-Based Vision Assessment**: Ensemble models for accurate visual field prediction.
- **Signal Processing**: Advanced feature extraction (FFT, jerk, gaze ratio, etc.).
- **Model Inference API**: Integrated with Flask and Streamlit for easy deployment.
- **Data Visualization**: Real-time analysis of gaze patterns and model outputs.

## 🧠 Technologies Used

- **Frontend:** Streamlit
- **Backend:** Flask API
- **ML Libraries:** scikit-learn, XGBoost, LightGBM
- **Computer Vision:** OpenCV, Dlib, Mediapipe
- **Data Handling:** Pandas, NumPy, SciPy
- **Deployment:** Docker (optional), Gunicorn

## ⚡ Installation

```bash
git clone https://github.com/Lig17/Enhancing-Central-Field-Visual-Assessment-Test-for-Pre-Verbal-and-Mentally-Challenged.git
cd Enhancing-Central-Field-Visual-Assessment-Test-for-Pre-Verbal-and-Mentally-Challenged
pip install -r requirements.txt
```

## 🚀 Run the Application

### 1️⃣ **Run Gaze Tracking & ML Models**

```bash
streamlit run streamlit_app.py
```

### 2️⃣ **Run Flask API for ML Inference**

```bash
python ml/api_integration.py
```

## 📊 Gaze Tracking Demo

1. Go to the **Gaze Tracker** tab in Streamlit.
2. Click **Start Gaze Tracking** to activate the webcam.
3. Eye Aspect Ratio (EAR) and Gaze Ratio will be displayed in real-time.
4. Click **Analyze Gaze Data** to get ML model predictions.

## 📦 Project Structure

```
/CFDVisual
├── ml/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── ensemble_models.py
│   ├── model_evaluation.py
│   ├── model_inference.py
│   ├── gaze_tracking_utils.py
│   └── api_integration.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## 🤖 ML Models Included

- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **XGBoost & LightGBM**
- **Stacking and Voting Ensembles**
- **Bagging Classifier**

## ✅ Sample Commands

### Run Unit Tests
```bash
pytest
```

### Build Docker Image (Optional)
```bash
docker build -t cfd-visual-app .
docker run -p 8501:8501 cfd-visual-app
```

## 📚 References
- [Dlib Documentation](http://dlib.net/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [OpenCV Documentation](https://opencv.org/)

## 🤝 Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

---

**© 2024 - Lig17 | Enhancing Vision Assessment for Pre-Verbal and Mentally Challenged Individuals**
