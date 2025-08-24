"""
Tests for the ZeroTune command-line interface using the new modular API.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from zerotune.__main__ import main


def test_cli_predict_openml():
    """Test CLI predict command with OpenML dataset."""
    with patch.object(sys, 'argv', ['zerotune', 'predict', '--dataset-id', '61', '--model-type', 'decision_tree']):
        with patch('zerotune.core.zero_tune.ZeroTune._optimize_single_dataset') as mock_optimize:
            mock_optimize.return_value = ({"max_depth": 10, "max_features": 0.7}, 0.85)
            
            exit_code = main()
            
            mock_optimize.assert_called_once()
            assert exit_code == 0


def test_cli_predict_custom(tmp_path):
    """Test CLI predict command with custom dataset file."""
    # Create a temporary CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6], 
        'target': [0, 1, 0]
    })
    df.to_csv(csv_path, index=False)
    
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['zerotune', 'predict', '--data-path', str(csv_path), '--target', 'target']
        
        with patch('zerotune.core.zero_tune.ZeroTune._optimize_single_dataset') as mock_optimize:
            mock_optimize.return_value = ({"max_depth": 10, "max_features": 0.7}, 0.85)
            
            exit_code = main()
            
            assert exit_code == 0
            mock_optimize.assert_called_once()
    finally:
        sys.argv = original_argv


def test_cli_error_handling():
    """Test CLI error handling when optimization fails."""
    with patch('zerotune.core.zero_tune.ZeroTune._optimize_single_dataset') as mock_optimize:
        mock_optimize.side_effect = ValueError("Test error")
        
        with patch.object(sys, 'argv', ['zerotune', 'predict', '--dataset-id', '1464']):
            exit_code = main()
        
        assert exit_code == 1


@pytest.fixture
def mock_get_ids():
    """Create a mock get_dataset_ids function."""
    with patch('zerotune.core.data_loading.get_dataset_ids') as mock:
        mock.return_value = [31, 44, 61]
        yield mock


def test_cli_datasets(mock_get_ids):
    """Test CLI datasets command."""
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['zerotune', 'datasets']
        
        with patch('zerotune.core.data_loading.load_dataset_catalog') as mock_load_catalog:
            mock_load_catalog.return_value = {
                'binary': [{'id': '1', 'n_classes': 2}],
                'multiclass': [{'id': '3', 'n_classes': 3}]
            }
            
            exit_code = main()
            
            assert exit_code == 0
            mock_get_ids.assert_called_once_with(category='all')
    finally:
        sys.argv = original_argv 