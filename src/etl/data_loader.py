import os
import pandas as pd
import numpy as np

def download_dataset(data_path):
    """Downloads the Credit Card Fraud dataset if not present."""
    import urllib.request
    import zipfile
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Multiple sources to try (the dataset is publicly available)
    urls = [
        # OpenML mirror of the dataset
        "https://www.openml.org/data/get_csv/1673544/phpKo8OWT",
    ]
    
    print("Downloading Credit Card Fraud dataset...")
    
    for url in urls:
        try:
            print(f"Trying: {url[:50]}...")
            urllib.request.urlretrieve(url, data_path)
            
            # Verify it's a valid CSV
            df = pd.read_csv(data_path, nrows=5)
            if 'Class' in df.columns or 'class' in df.columns.str.lower():
                print(f"Successfully downloaded dataset to {data_path}")
                return True
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    print("Could not download from automatic sources.")
    print("Please download manually from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print(f"And place 'creditcard.csv' in: {os.path.dirname(data_path)}")
    return False

def load_data(config):
    """Loads the dataset from the specified path or downloads/generates it."""
    data_path = config['data']['raw_data_path']
    
    # Try to load existing data
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            print("Loaded real Kaggle dataset!")
            return data
        except Exception as e:
            print(f"Error reading dataset: {e}")
    
    # Try to download the dataset
    print(f"Dataset not found at: {data_path}")
    if download_dataset(data_path):
        try:
            data = pd.read_csv(data_path)
            print("Loaded downloaded dataset!")
            return data
        except Exception as e:
            print(f"Error reading downloaded dataset: {e}")
    
    # Fall back to synthetic data if enabled
    if config['synthetic_data']['generate']:
        print("Falling back to synthetic fraud dataset...")
        return generate_synthetic_data(config)
    
    print("Real dataset not found and synthetic data generation is disabled.")
    return None

def generate_synthetic_data(config):
    """Generates a synthetic dataset for fraud detection."""
    np.random.seed(config['data_processing']['random_state'])
    n_samples = config['synthetic_data']['n_samples']
    fraud_rate = config['synthetic_data']['fraud_rate']

    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    normal_features = np.random.normal(0, 1, (n_normal, 28))
    fraud_features = np.random.normal(1.5, 1.2, (n_fraud, 28))

    normal_time = np.random.uniform(0, 172800, n_normal)
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    normal_amount = np.random.lognormal(3, 1.5, n_normal)
    fraud_amount = np.random.lognormal(5, 2, n_fraud)

    normal_data = np.column_stack([normal_features, normal_time, normal_amount])
    fraud_data = np.column_stack([fraud_features, fraud_time, fraud_amount])

    X = np.vstack([normal_data, fraud_data])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    data = pd.DataFrame(X, columns=columns)
    data['Class'] = y

    return data
