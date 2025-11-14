import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.etl.data_loader import load_data
from src.utils.utils import load_config

def test_load_data():
    """Tests the load_data function."""
    config = load_config()
    config['synthetic_data']['generate'] = True
    data = load_data(config)

    assert data is not None
    assert 'Class' in data.columns
    assert len(data) == config['synthetic_data']['n_samples']
