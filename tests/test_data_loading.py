"""
Tests for the data loading module in ZeroTune.
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from zerotune.core.data_loading import (
    prepare_data,
    load_dataset_catalog,
    get_dataset_ids,
    get_recommended_datasets,
    fetch_open_ml_data
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    # Create a small dataframe with numerical and categorical columns
    data = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature3': ['a', 'b', 'c', 'a', 'b'],
        'target': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataset_with_nans():
    """Create a sample dataset with missing values."""
    # Create a small dataframe with NaN values
    data = {
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [0.1, np.nan, 0.3, 0.4, 0.5],
        'feature3': ['a', 'b', 'c', np.nan, 'b'],
        'target': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_catalog():
    """Create a sample dataset catalog for testing."""
    return {
        "binary": [
            {"id": "31", "n_classes": 2, "name": "credit-g"},
            {"id": "44", "n_classes": 2, "name": "spambase"},
            {"id": "61", "n_classes": 2, "name": "iris"}
        ],
        "multiclass": [
            {"id": "38", "n_classes": 3, "name": "iris"},
            {"id": "42", "n_classes": 4, "name": "car"},
            {"id": "50", "n_classes": 5, "name": "wine"}
        ]
    }


@pytest.fixture
def mock_dataset():
    """Create a mock OpenML dataset."""
    mock = MagicMock()
    mock.name = "mock_dataset"
    mock.default_target_attribute = "target"
    mock.get_data.return_value = (
        pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
        pd.Series([0, 1, 0]),
        [False, False],  # categorical_indicator
        ['feature1', 'feature2']  # attribute_names
    )
    return mock


def test_prepare_data(sample_dataset, sample_dataset_with_nans):
    """Test the prepare_data function."""
    # Test with a clean dataset
    X, y = prepare_data(sample_dataset, 'target')
    
    # Check shape
    assert X.shape[0] == 5
    assert X.shape[1] == 3
    assert y.shape[0] == 5
    
    # Check column names
    assert 'feature1' in X.columns
    assert 'feature2' in X.columns
    assert 'feature3' in X.columns
    assert 'target' not in X.columns
    
    # Test with a dataset containing NaNs
    X_with_nans, y_with_nans = prepare_data(sample_dataset_with_nans, 'target')
    
    # Check that NaNs were filled
    assert not X_with_nans.isna().any().any()
    
    # Check specific values
    assert X_with_nans.loc[2, 'feature1'] == 0  # NaN should be filled with 0
    assert X_with_nans.loc[1, 'feature2'] == 0  # NaN should be filled with 0


def test_load_dataset_catalog():
    """Test the load_dataset_catalog function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample catalog file
        catalog_path = os.path.join(temp_dir, "test_catalog.json")
        sample_catalog = {
            "binary": {"dataset1": 1, "dataset2": 2},
            "multi-class": {"dataset3": 3, "dataset4": 4}
        }
        
        # Write the catalog to a file
        with open(catalog_path, 'w') as f:
            json.dump(sample_catalog, f)
        
        # Test loading the catalog
        loaded_catalog = load_dataset_catalog(catalog_path)
        assert "binary" in loaded_catalog
        assert "multi-class" in loaded_catalog
        assert loaded_catalog["binary"]["dataset1"] == 1
        assert loaded_catalog["multi-class"]["dataset3"] == 3
        
        # Test loading a non-existent file
        empty_catalog = load_dataset_catalog(os.path.join(temp_dir, "nonexistent.json"))
        assert "binary" in empty_catalog
        assert "multi-class" in empty_catalog
        assert empty_catalog["binary"] == {}
        assert empty_catalog["multi-class"] == {}


def test_fetch_open_ml_data_mock(mock_dataset, monkeypatch):
    """Test fetching OpenML data with a mock."""
    # Mock the fetch_openml function
    def mock_fetch(*args, **kwargs):
        return mock_dataset
    
    monkeypatch.setattr("openml.datasets.get_dataset", mock_fetch)
    
    # Test the function
    df, target_name, dataset_name = fetch_open_ml_data(123)
    
    # Verify the results
    assert isinstance(df, pd.DataFrame)
    assert target_name == 'target'
    assert dataset_name == "mock_dataset"
    assert list(df.columns) == ['feature1', 'feature2', 'target']
    assert len(df) == 3 