import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from feature_engineering import engineer_features

def load_and_prepare_data(filepath):
    features, labels = engineer_features(filepath)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def initialize_models():
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

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions))
        joblib.dump(model, f'{name.lower()}_model.pkl')
    return results

def create_ensemble_model(models, X_train, y_train):
    estimators = [(name, model) for name, model in models.items() if name not in ['NaiveBayes', 'KNN']]
    meta_model = LogisticRegression()

    stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5)
    stacking_model.fit(X_train, y_train)
    joblib.dump(stacking_model, 'stacking_model.pkl')
    return stacking_model

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data('gaze_data.csv')
    models = initialize_models()
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)

    # Ensemble Model
    ensemble_model = create_ensemble_model(models, X_train, y_train)
    ensemble_predictions = ensemble_model.predict(X_test)
    print("Ensemble Model Accuracy:", accuracy_score(y_test, ensemble_predictions))

    # Hyperparameter Tuning
    best_rf_model = hyperparameter_tuning(X_train, y_train)
    tuned_predictions = best_rf_model.predict(X_test)
    print("Tuned RandomForest Accuracy:", accuracy_score(y_test, tuned_predictions))
