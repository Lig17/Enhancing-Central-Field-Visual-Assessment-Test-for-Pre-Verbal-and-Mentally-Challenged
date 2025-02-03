
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import skew, kurtosis

def extract_temporal_features(data):
    data['velocity'] = np.gradient(data['gaze_position'])
    data['acceleration'] = np.gradient(data['velocity'])
    data['jerk'] = np.gradient(data['acceleration'])

    data['savgol_velocity'] = savgol_filter(data['velocity'], window_length=5, polyorder=2)
    return data

def extract_statistical_features(data):
    data['mean_gaze'] = data['gaze_position'].rolling(window=5).mean().fillna(method='bfill')
    data['std_gaze'] = data['gaze_position'].rolling(window=5).std().fillna(method='bfill')
    data['skew_gaze'] = data['gaze_position'].rolling(window=5).apply(skew).fillna(method='bfill')
    data['kurtosis_gaze'] = data['gaze_position'].rolling(window=5).apply(kurtosis).fillna(method='bfill')
    return data

def extract_frequency_features(data):
    fft_values = np.fft.fft(data['gaze_position'].fillna(0))
    fft_magnitude = np.abs(fft_values)

    data['fft_max'] = np.max(fft_magnitude)
    data['fft_mean'] = np.mean(fft_magnitude)
    data['fft_std'] = np.std(fft_magnitude)
    return data

def extract_peak_features(data):
    peaks, _ = find_peaks(data['gaze_position'], height=0)
    data['num_peaks'] = len(peaks)
    data['peak_mean'] = np.mean(data['gaze_position'].iloc[peaks]) if peaks.size > 0 else 0
    data['peak_std'] = np.std(data['gaze_position'].iloc[peaks]) if peaks.size > 0 else 0
    return data

def engineer_features(filepath):
    data = pd.read_csv(filepath)
    data = extract_temporal_features(data)
    data = extract_statistical_features(data)
    data = extract_frequency_features(data)
    data = extract_peak_features(data)
    
    # Selecting relevant features
    feature_columns = [
        'gaze_position', 'velocity', 'acceleration', 'jerk', 'savgol_velocity',
        'mean_gaze', 'std_gaze', 'skew_gaze', 'kurtosis_gaze',
        'fft_max', 'fft_mean', 'fft_std',
        'num_peaks', 'peak_mean', 'peak_std'
    ]

    features = data[feature_columns]
    labels = data['vision_class'] if 'vision_class' in data.columns else None
    return features, labels

if __name__ == "__main__":
    features, labels = engineer_features('gaze_data.csv')
    print(features.head())
