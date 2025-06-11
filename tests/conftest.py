"""
Common test fixtures and configurations for ZeroTune tests.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import tempfile
import joblib
from sklearn.ensemble import RandomForestRegressor

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
def empty_dataset():
    """
    Fixture that creates an empty dataset for edge case testing.
    
    This fixture creates a DataFrame and Series with 0 rows to test
    how functions handle empty data.
    """
    columns = [f'feature_{i}' for i in range(5)]
    X = pd.DataFrame(columns=columns)
    y = pd.Series([], name='target')
    return X, y


@pytest.fixture
def single_row_dataset():
    """
    Fixture that creates a dataset with only one row for edge case testing.
    
    This is useful for testing minimum dataset size requirements and
    statistical calculations that typically need more than one data point.
    """
    X = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5]], columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series([1], name='target')
    return X, y


@pytest.fixture
def high_dimensional_dataset():
    """
    Fixture that creates a high-dimensional dataset (many features).
    
    This helps test the system's capability to handle datasets with a large
    number of features, which can challenge computational efficiency and
    potentially reveal scaling issues.
    """
    X, y = make_classification(
        n_samples=50,
        n_features=100,  # High number of features
        n_informative=10,
        n_redundant=20,
        n_repeated=5,
        n_classes=2,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='target')


@pytest.fixture
def all_categorical_dataset():
    """
    Fixture that creates a dataset with all categorical features.
    
    This tests the system's ability to handle categorical data appropriately.
    The dataset includes both binary and multi-category features.
    """
    # Create categorical data
    n_samples = 100
    np.random.seed(42)
    
    # Binary features
    binary_features = np.random.randint(0, 2, size=(n_samples, 3))
    
    # Multi-category features
    multi_cat1 = np.random.randint(0, 3, size=n_samples)
    multi_cat2 = np.random.randint(0, 5, size=n_samples)
    
    # Combine features
    X_data = np.column_stack([binary_features, multi_cat1.reshape(-1, 1), multi_cat2.reshape(-1, 1)])
    
    # Create DataFrame with categorical dtypes
    X = pd.DataFrame(X_data, columns=['binary1', 'binary2', 'binary3', 'category3', 'category5'])
    for col in X.columns:
        X[col] = X[col].astype('category')
    
    # Create categorical target
    y = pd.Series(np.random.randint(0, 2, size=n_samples), name='target').astype('category')
    
    return X, y


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


@pytest.fixture
def synthetic_dataset():
    """Generate a synthetic classification dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    # Convert to pandas DataFrame/Series for compatibility
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def dataset_meta_params():
    """Create a sample set of dataset meta-parameters for testing."""
    return {
        "n_samples": 1000,
        "n_features": 20,
        "n_highly_target_corr": 5,
        "imbalance_ratio": 2.0,
        "avg_feature_m3": 0.2,
        "var_feature_m3": 0.1,
        "avg_feature_m1": 0.5,
        "var_feature_m1": 0.3,
        "avg_feature_m2": 1.2,
        "var_feature_m2": 0.8,
        "avg_feature_m4": 3.0,
        "var_feature_m4": 1.5,
        "avg_target_corr": 0.3,
        "var_target_corr": 0.2
    }


@pytest.fixture
def mock_kb_data():
    """Create mock knowledge base data for testing."""
    # Create dataset features dataframe
    df_features = pd.DataFrame({
        "Dataset": [1, 2, 3],
        "n_samples": [100, 200, 300],
        "n_features": [10, 20, 30],
        "n_highly_target_corr": [3, 5, 7],
        "imbalance_ratio": [1.5, 2.0, 3.0],
        "avg_feature_m3": [0.1, 0.2, 0.3],
        "var_feature_m3": [0.05, 0.1, 0.15]
    })
    
    # Create trials dataframe
    df_trials = pd.DataFrame({
        "Dataset": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "value": [0.8, 0.7, 0.9, 0.75, 0.85, 0.8, 0.9, 0.85, 0.95],
        "params_max_depth": [5, 10, 15, 8, 12, 10, 10, 15, 20],
        "params_max_features": [0.5, 0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.8, 0.9],
        "params_min_samples_leaf": [1, 2, 1, 2, 1, 2, 1, 1, 2],
        "params_min_samples_split": [2, 5, 2, 5, 2, 5, 2, 5, 2],
        "rank": [2, 3, 1, 3, 1, 2, 2, 3, 1]
    })
    
    return df_features, df_trials


@pytest.fixture
def mock_model_path():
    """Create a temporary model file for testing prediction functions."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a simple model 
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.rand(10, 4)  # 4 outputs for the hyperparameters
        model.fit(X, y)
        
        # Create a directory for models
        os.makedirs(os.path.join(tmpdirname, "pretrained_models"), exist_ok=True)
        model_path = os.path.join(tmpdirname, "pretrained_models", "test_model.joblib")
        
        # Save the model
        joblib.dump(model, model_path)
        
        yield model_path


@pytest.fixture
def decision_tree_config():
    """Sample decision tree configuration for testing."""
    return {
        "name": "DecisionTreeClassifier",
        "param_config": {
            "max_depth": {
                "percentage_splits": [0.25, 0.50, 0.70, 0.8, 0.9, 0.999],
                "param_type": "int",
                "dependency": "n_samples"
            },
            "min_samples_split": {
                "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.10],
                "param_type": "float"
            },
            "min_samples_leaf": {
                "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.10],
                "param_type": "float"
            },
            "max_features": {
                "percentage_splits": [0.50, 0.70, 0.8, 0.9, 0.99],
                "param_type": "float"
            }
        }
    }


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create subdirectories that might be needed
        os.makedirs(os.path.join(tmpdirname, "pretrained_models"), exist_ok=True)
        os.makedirs(os.path.join(tmpdirname, "checkpoint"), exist_ok=True)
        yield tmpdirname 