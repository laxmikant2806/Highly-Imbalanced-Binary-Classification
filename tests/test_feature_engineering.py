import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.features.feature_engineering import split_data, scale_features
from src.utils.utils import load_config

@pytest.fixture
def sample_data():
    """Provides sample data for testing."""
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_split_data(sample_data):
    """Tests the split_data function."""
    X, y = sample_data
    config = load_config()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)

def test_scale_features(sample_data):
    """Tests the scale_features function."""
    X, y = sample_data
    config = load_config()
    X_train, X_val, X_test, _, _, _ = split_data(X, y, config)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape == X_val.shape
    assert X_test_scaled.shape == X_test.shape
    assert scaler is not None
