import numpy as np
import pandas as pd
import logging
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import skew, kurtosis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_temporal_features(data):
    """Extracts velocity, acceleration, jerk, and smooth velocity."""
    if 'gaze_position' not in data.columns:
        logging.error("Missing 'gaze_position' column in data.")
        raise ValueError("Data must contain 'gaze_position' column.")

    data['velocity'] = np.gradient(data['gaze_position'].fillna(0))
    data['acceleration'] = np.gradient(data['velocity'])
    data['jerk'] = np.gradient(data['acceleration'])

    # Apply Savitzky-Golay smoothing filter
    data['savgol_velocity'] = savgol_filter(data['velocity'], window_length=5, polyorder=2, mode='nearest')

    return data

def extract_statistical_features(data):
    """Extracts statistical features: mean, std, skewness, and kurtosis."""
    rolling_window = 5
    data['mean_gaze'] = data['gaze_position'].rolling(window=rolling_window, min_periods=1).mean()
    data['std_gaze'] = data['gaze_position'].rolling(window=rolling_window, min_periods=1).std()
    data['skew_gaze'] = data['gaze_position'].rolling(window=rolling_window, min_periods=1).apply(skew, raw=True)
    data['kurtosis_gaze'] = data['gaze_position'].rolling(window=rolling_window, min_periods=1).apply(kurtosis, raw=True)

    return data.fillna(method='bfill')  # Fill missing values

def extract_frequency_features(data):
    """Extracts frequency-domain features using FFT."""
    if 'gaze_position' not in data.columns:
        raise ValueError("Data must contain 'gaze_position' column.")

    fft_values = np.fft.fft(data['gaze_position'].fillna(0).values)
    fft_magnitude = np.abs(fft_values)

    data['fft_max'] = np.max(fft_magnitude)
    data['fft_mean'] = np.mean(fft_magnitude)
    data['fft_std'] = np.std(fft_magnitude)

    return data

def extract_peak_features(data):
    """Extracts peak-based features."""
    peaks, _ = find_peaks(data['gaze_position'].fillna(0), height=0)
    
    data['num_peaks'] = len(peaks)
    data['peak_mean'] = np.mean(data['gaze_position'].iloc[peaks]) if peaks.size > 0 else 0
    data['peak_std'] = np.std(data['gaze_position'].iloc[peaks]) if peaks.size > 0 else 0

    return data

def engineer_features(data_input):
    """
    Processes gaze data to extract engineered features.
    
    Args:
        data_input (str or pd.DataFrame): File path or DataFrame containing gaze data.
    
    Returns:
        pd.DataFrame: Extracted features.
        pd.Series or None: Labels (if available).
    """
    # Load data if file path is given
    if isinstance(data_input, str):
        if not data_input.endswith('.csv'):
            logging.error("Only CSV files are supported.")
            raise ValueError("Invalid file format. Provide a CSV file.")
        
        data = pd.read_csv(data_input)
        logging.info(f"Loaded data from {data_input} with shape {data.shape}")
    elif isinstance(data_input, pd.DataFrame):
        data = data_input.copy()
        logging.info(f"Processing DataFrame with shape {data.shape}")
    else:
        raise TypeError("Input must be a file path or a pandas DataFrame.")

    # Extract Features
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
    
    features = data[feature_columns].copy()

    # Extract labels if available
    labels = data['vision_class'] if 'vision_class' in data.columns else None

    logging.info(f"Feature extraction complete. Feature shape: {features.shape}")
    return features, labels

if __name__ == "__main__":
    features, labels = engineer_features('gaze_data.csv')
    print(features.head())
