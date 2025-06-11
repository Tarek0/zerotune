"""
Tests for the feature extraction module in ZeroTune.
"""

import pytest
import pandas as pd
import numpy as np

from zerotune.core.feature_extraction import (
    _calculate_dataset_size,
    _calculate_class_imbalance_ratio,
    _calculate_correlation_metrics,
    _calculate_feature_moments_and_variances,
    _calculate_row_moments_and_variances,
    calculate_dataset_meta_parameters
)


@pytest.fixture
def synthetic_dataset():
    """Create a synthetic dataset for testing."""
    from sklearn.datasets import make_classification
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5,
        n_redundant=3, n_classes=2, random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


def test_calculate_dataset_size(synthetic_dataset):
    """Test the _calculate_dataset_size function."""
    X, _ = synthetic_dataset
    
    # Calculate dataset size
    result = _calculate_dataset_size(X)
    
    # Check the result has the expected keys and values
    assert "n_samples" in result
    assert "n_features" in result
    assert result["n_samples"] == 100
    assert result["n_features"] == 10


def test_calculate_class_imbalance_ratio():
    """Test the _calculate_class_imbalance_ratio function."""
    # Create test data with imbalanced classes
    y_balanced = np.array([0, 0, 1, 1])  # Balanced: 2:2 = 1.0
    y_imbalanced = np.array([0, 0, 0, 1])  # Imbalanced: 3:1 = 3.0
    
    # Calculate imbalance ratios
    result_balanced = _calculate_class_imbalance_ratio(y_balanced)
    result_imbalanced = _calculate_class_imbalance_ratio(y_imbalanced)
    
    # Check results
    assert "imbalance_ratio" in result_balanced
    assert "imbalance_ratio" in result_imbalanced
    assert result_balanced["imbalance_ratio"] == 1.0
    assert result_imbalanced["imbalance_ratio"] == 3.0


def test_calculate_correlation_metrics(synthetic_dataset):
    """Test the _calculate_correlation_metrics function."""
    X, y = synthetic_dataset
    
    # Calculate correlation metrics
    result = _calculate_correlation_metrics(X, y)
    
    # Check the result has the expected keys
    assert "n_highly_target_corr" in result
    assert "avg_target_corr" in result
    assert "var_target_corr" in result
    
    # Check types
    assert isinstance(result["n_highly_target_corr"], int)
    assert isinstance(result["avg_target_corr"], float)
    assert isinstance(result["var_target_corr"], float)
    
    # Test with a different correlation cutoff
    result_high_cutoff = _calculate_correlation_metrics(X, y, correlation_cutoff=0.5)
    
    # The number of highly correlated features should be lower with a higher cutoff
    assert result_high_cutoff["n_highly_target_corr"] <= result["n_highly_target_corr"]


def test_calculate_feature_moments_and_variances(synthetic_dataset):
    """Test the _calculate_feature_moments_and_variances function."""
    X, _ = synthetic_dataset
    
    # Calculate feature moments
    result = _calculate_feature_moments_and_variances(X)
    
    # Check the result has the expected keys
    expected_keys = [
        'avg_feature_m1', 'var_feature_m1',
        'avg_feature_m2', 'var_feature_m2',
        'avg_feature_m3', 'var_feature_m3',
        'avg_feature_m4', 'var_feature_m4'
    ]
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], float)


def test_calculate_row_moments_and_variances(synthetic_dataset):
    """Test the _calculate_row_moments_and_variances function."""
    X, _ = synthetic_dataset
    
    # Calculate row moments
    result = _calculate_row_moments_and_variances(X)
    
    # Check the result has the expected keys
    expected_keys = [
        'avg_row_m1', 'var_row_m1',
        'avg_row_m2', 'var_row_m2',
        'avg_row_m3', 'var_row_m3',
        'avg_row_m4', 'var_row_m4'
    ]
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], float)


def test_calculate_dataset_meta_parameters(synthetic_dataset):
    """Test the calculate_dataset_meta_parameters function."""
    X, y = synthetic_dataset
    
    # Calculate all meta-parameters
    result = calculate_dataset_meta_parameters(X, y)
    
    # Check that the result combines outputs from all sub-functions
    assert "n_samples" in result  # From _calculate_dataset_size
    assert "imbalance_ratio" in result  # From _calculate_class_imbalance_ratio
    assert "n_highly_target_corr" in result  # From _calculate_correlation_metrics
    assert "avg_feature_m1" in result  # From _calculate_feature_moments_and_variances
    assert "avg_row_m1" in result  # From _calculate_row_moments_and_variances
    
    # Count the number of meta-parameters
    # We should have at least 15 meta-parameters
    assert len(result) >= 15 