import pandas as pd
import numpy as np

def load_data(config):
    """Loads the dataset from the specified path or generates synthetic data."""
    try:
        data = pd.read_csv(config['data']['raw_data_path'])
        print("Loaded real Kaggle dataset!")
    except FileNotFoundError:
        if not config['synthetic_data']['generate']:
            print("Real dataset not found and synthetic data generation is disabled.")
            return None

        print("Creating synthetic fraud dataset...")
        data = generate_synthetic_data(config)

    return data

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
