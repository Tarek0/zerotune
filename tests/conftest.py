"""
Common test fixtures and configurations for ZeroTune tests.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Add the parent directory to the path so we can import zerotune
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def small_classification_dataset():
    """Fixture that creates a small classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='target')


@pytest.fixture
def small_regression_dataset():
    """Fixture that creates a small regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='target')


@pytest.fixture
def mock_dataset_meta_parameters():
    """Fixture that returns mock dataset meta parameters."""
    return {
        'n_samples': 100,
        'n_features': 5,
        'n_classes': 2,
        'n_numeric': 5,
        'n_categorical': 0,
        'n_binary': 0,
        'target_type': 'classification',
        'imbalance_ratio': 1.0,
        'n_highly_target_corr': 2,
        'feature_correlation_avg': 0.2,
        'target_entropy': 0.9
    }


@pytest.fixture
def decision_tree_param_config():
    """Fixture that returns a parameter configuration for Decision Tree."""
    return {
        "max_depth": {
            "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999],
            "param_type": "int",
            "dependency": "n_samples"
        },
        "min_samples_split": {
            "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1],
            "param_type": "float"
        },
        "min_samples_leaf": {
            "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1],
            "param_type": "float"
        },
        "max_features": {
            "percentage_splits": [0.5, 0.7, 0.8, 0.9, 0.99],
            "param_type": "float"
        }
    } 