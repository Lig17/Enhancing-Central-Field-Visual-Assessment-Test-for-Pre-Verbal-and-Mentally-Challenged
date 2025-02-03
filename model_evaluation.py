import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_test_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('vision_class', axis=1)
    y = data['vision_class']
    return X, y

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    roc_auc = roc_auc_score(y_test, probabilities) if probabilities is not None else 'N/A'

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if probabilities is not None:
        print(f"ROC AUC Score: {roc_auc:.4f}")

    print("Classification Report:\n", classification_report(y_test, predictions))

    plot_confusion_matrix(y_test, predictions)
    if probabilities is not None:
        plot_roc_curve(y_test, probabilities)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    X_test, y_test = load_test_data('test_gaze_data.csv')
    models = [
        'rf_model.pkl',
        'gb_model.pkl',
        'xgboost_model.pkl',
        'lightgbm_model.pkl',
        'stacking_model.pkl',
        'voting_model.pkl',
        'bagging_model.pkl'
    ]

    for model_path in models:
        model = joblib.load(model_path)
        print(f"\nEvaluating Model: {model_path}")
        evaluate_model(model, X_test, y_test)