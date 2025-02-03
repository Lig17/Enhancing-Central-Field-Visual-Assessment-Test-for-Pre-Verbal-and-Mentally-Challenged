
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    data.fillna(method='ffill', inplace=True)

    data['velocity'] = np.gradient(data['gaze_position'])
    data['acceleration'] = np.gradient(data['velocity'])
    data['jerk'] = np.gradient(data['acceleration'])

    data['mean_gaze'] = data['gaze_position'].rolling(window=5).mean().fillna(method='bfill')
    data['std_gaze'] = data['gaze_position'].rolling(window=5).std().fillna(method='bfill')

    
    features = data[['gaze_position', 'velocity', 'acceleration', 'jerk', 'mean_gaze', 'std_gaze']]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    labels = data['vision_class']
    return train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    data = load_data('gaze_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
