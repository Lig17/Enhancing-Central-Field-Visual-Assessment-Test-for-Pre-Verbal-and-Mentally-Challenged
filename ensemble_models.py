import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from feature_engineering import engineer_features

def load_data(filepath):
    features, labels = engineer_features(filepath)
    return train_test_split(features, labels, test_size=0.2, random_state=42)

def initialize_base_models():
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_leaves=31, learning_rate=0.05),
        'SVM': SVC(kernel='rbf', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'RidgeClassifier': RidgeClassifier()
    }
    return models

def create_stacking_ensemble(models, X_train, y_train):
    estimators = [(name, model) for name, model in models.items() if name not in ['NaiveBayes', 'KNN']]
    meta_model = LogisticRegression()

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        passthrough=True
    )

    stacking_model.fit(X_train, y_train)
    joblib.dump(stacking_model, 'stacking_model.pkl')
    return stacking_model

def create_voting_ensemble(models, X_train, y_train):
    voting_model = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)
    joblib.dump(voting_model, 'voting_model.pkl')
    return voting_model

def create_bagging_ensemble(base_model, X_train, y_train):
    bagging_model = BaggingClassifier(base_estimator=base_model, n_estimators=50, random_state=42)
    bagging_model.fit(X_train, y_train)
    joblib.dump(bagging_model, 'bagging_model.pkl')
    return bagging_model

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{matrix}")

def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('gaze_data.csv')
    models = initialize_base_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name)
        cross_validate_model(model, X_train, y_train)

    stacking_model = create_stacking_ensemble(models, X_train, y_train)
    evaluate_model(stacking_model, X_test, y_test, "Stacking Ensemble")
    cross_validate_model(stacking_model, X_train, y_train)

    voting_model = create_voting_ensemble(models, X_train, y_train)
