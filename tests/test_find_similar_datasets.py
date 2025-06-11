"""
Tests for the similarity functions from optimization module.
"""

import pytest
import numpy as np
import pandas as pd
from zerotune.core.optimization import find_similar_datasets


@pytest.fixture
def synthetic_meta_parameters():
    """Create synthetic meta-parameters for testing."""
    return {
        "n_samples": 100,
        "n_features": 10,
        "imbalance_ratio": 1.0,
        "n_highly_target_corr": 3,
        "avg_target_corr": 0.25,
        "var_target_corr": 0.05,
        "avg_feature_m1": 0.0,
        "var_feature_m1": 1.0,
        "avg_feature_m2": 1.0,
        "var_feature_m2": 0.1,
        "avg_feature_m3": 0.0,
        "var_feature_m3": 0.2,
        "avg_feature_m4": 3.0,
        "var_feature_m4": 0.5,
        "avg_row_m1": 0.0,
        "var_row_m1": 0.3,
        "avg_row_m2": 1.0,
        "var_row_m2": 0.2,
        "avg_row_m3": 0.0,
        "var_row_m3": 0.1,
        "avg_row_m4": 3.0,
        "var_row_m4": 0.4
    }


def test_find_similar_datasets(synthetic_meta_parameters):
    """Test finding similar datasets based on meta-features."""
    # Create a knowledge base with multiple entries
    kb = {
        "meta_features": [
            # Base entry (very similar to synthetic_meta_parameters)
            {
                "n_samples": 100,
                "n_features": 10,
                "imbalance_ratio": 1.0,
                "avg_feature_m1": 0.0,
                "var_feature_m1": 1.0
            },
            # Small dataset (somewhat similar)
            {
                "n_samples": 50,
                "n_features": 5,
                "imbalance_ratio": 1.2,
                "avg_feature_m1": 0.1,
                "var_feature_m1": 0.9
            },
            # Large dataset (not similar)
            {
                "n_samples": 1000,
                "n_features": 50,
                "imbalance_ratio": 10.0,
                "avg_feature_m1": 0.5,
                "var_feature_m1": 0.5
            }
        ]
    }
    
    # Find similar datasets
    similar_indices = find_similar_datasets(
        synthetic_meta_parameters,
        kb,
        n_neighbors=2
    )
    
    # Verify the result
    assert len(similar_indices) == 2
    # The most similar dataset should be the first one (index 0)
    assert 0 in similar_indices
    
    # Test with empty knowledge base
    empty_kb = {"meta_features": []}
    empty_result = find_similar_datasets(
        synthetic_meta_parameters,
        empty_kb,
        n_neighbors=1
    )
    assert empty_result == [] 