"""
Tests for the ZeroTune command-line interface using the new modular API.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import io
import contextlib

from zerotune.__main__ import main
from zerotune.core.data_loading import fetch_open_ml_data
from zerotune.core.zero_tune import ZeroTune


def test_cli_predict_openml():
    """Test CLI predict command with OpenML dataset ID using minimal mocking."""
    # We need to mock sys.argv to simulate CLI arguments
    # This is unavoidable for CLI testing
    with patch.object(sys, 'argv', ['zerotune', 'predict', '--dataset', '61', '--model', 'decision_tree']):
        # We'll run with a real but small dataset (iris) and minimal iterations
        with patch('zerotune.core.zero_tune.ZeroTune.optimize') as mock_optimize:
            # We only mock the optimize method to avoid actual computation,
            # but we'll verify it's called with correct parameters
            mock_optimize.return_value = (
                {"max_depth": 10, "max_features": 0.7}, 
                0.85, 
                MagicMock()
            )
            
            # Run the command
            exit_code = main()
            
            # Check that ZeroTune's optimize was called with expected parameters
            mock_optimize.assert_called_once()
            
            # Check dataset_id was passed correctly from CLI args
            args, kwargs = mock_optimize.call_args
            assert kwargs.get('n_iter', 0) > 0  # Should have some iterations
            assert kwargs.get('verbose', False) == True  # Default is verbose output
            
            # Check exit code
            assert exit_code == 0


@pytest.fixture
def mock_zerotune():
    """Create a mock ZeroTune instance."""
    with patch('zerotune.__main__.ZeroTune') as mock:
        instance = mock.return_value

        # Create a mock model with fit/predict methods
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        # The predict method should return an array of the same length as the input `X`.
        mock_model.predict.side_effect = lambda X: [0] * len(X)

        instance.optimize.return_value = (
            {'max_depth': 5, 'min_samples_split': 2},
            0.95,
            mock_model
        )
        yield mock

@pytest.fixture
def mock_get_ids():
    """Create a mock get_dataset_ids function."""
    with patch('zerotune.core.data_loading.get_dataset_ids') as mock:
        mock.return_value = [31, 44, 61]
        yield mock

def test_cli_predict_custom(mock_zerotune, tmp_path):
    """Test CLI predict command with custom dataset."""
    # Create a temporary CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    df.to_csv(csv_path, index=False)
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    try:
        # Set up command line arguments
        sys.argv = ['zerotune', 'predict', '--data-path', str(csv_path), '--target', 'target']
        
        # Run the command
        exit_code = main()
        
        # Verify the results
        assert exit_code == 0
        mock_zerotune.assert_called_once()
        mock_zerotune.return_value.optimize.assert_called_once()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def test_cli_train(mock_zerotune):
    """Test CLI train command."""
    # Save original sys.argv
    original_argv = sys.argv.copy()
    try:
        # Set up command line arguments
        sys.argv = ['zerotune', 'train', '--dataset-id', '31']
        
        # Run the command
        exit_code = main()
        
        # Verify the results
        assert exit_code == 0
        mock_zerotune.assert_called_once()
        mock_zerotune.return_value.optimize.assert_called_once()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def test_cli_demo():
    """Test CLI demo command with minimal mocking."""
    # Mock the ZeroTune class to avoid actual computation
    with patch('zerotune.__main__.ZeroTune') as mock_zerotune:
        # Setup mock instance
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = (
            {"max_depth": 10, "max_features": 0.7}, 
            0.85, 
            MagicMock()
        )
        mock_zerotune.return_value = mock_instance
        
        # Run the command with arguments to test argument parsing
        with patch.object(sys, 'argv', ['zerotune', 'demo', '--model', 'decision_tree']):
            exit_code = main()
        
        # Check that ZeroTune was initialized and optimize was called
        mock_zerotune.assert_called_once()
        assert mock_instance.optimize.called
        
        # Verify model type was passed correctly
        args, kwargs = mock_zerotune.call_args
        assert kwargs.get('model_type') == 'decision_tree'
        
        # Check exit code
        assert exit_code == 0


@patch('zerotune.__main__.ZeroTune')
def test_cli_error_handling(mock_zerotune):
    """Test CLI error handling."""
    # Setup mock to raise an exception
    mock_instance = MagicMock()
    mock_instance.optimize.side_effect = ValueError("Test error")
    mock_zerotune.return_value = mock_instance
    
    # Run the command
    with patch.object(sys, 'argv', ['zerotune', 'predict', '--dataset', '1464']):
        exit_code = main()
    
    # Check that the error was caught and handled
    assert exit_code == 1


def test_cli_datasets(mock_get_ids):
    """Test CLI datasets command."""
    # Save original sys.argv
    original_argv = sys.argv.copy()
    try:
        # Set up command line arguments
        sys.argv = ['zerotune', 'datasets']
        
        # Mock the load_dataset_catalog function
        with patch('zerotune.core.data_loading.load_dataset_catalog') as mock_load_catalog:
            mock_load_catalog.return_value = {
                'binary': [{'id': '1', 'n_classes': 2}, {'id': '2', 'n_classes': 2}],
                'multiclass': [{'id': '3', 'n_classes': 3}, {'id': '4', 'n_classes': 4}]
            }
            
            # Run the command
            exit_code = main()
            
            # Verify the results
            assert exit_code == 0
            mock_get_ids.assert_called_once_with(category='all')
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def test_cli_datasets_list(mock_get_ids):
    """Test CLI datasets list command."""
    # Save original sys.argv
    original_argv = sys.argv.copy()
    try:
        # Set up command line arguments
        sys.argv = ['zerotune', 'datasets', '--category', 'binary']
        
        # Mock the load_dataset_catalog function
        with patch('zerotune.core.data_loading.load_dataset_catalog') as mock_load_catalog:
            mock_load_catalog.return_value = {
                'binary': [{'id': '1', 'n_classes': 2}, {'id': '2', 'n_classes': 2}],
                'multiclass': [{'id': '3', 'n_classes': 3}, {'id': '4', 'n_classes': 4}]
            }
            
            # Run the command
            exit_code = main()
            
            # Verify the results
            assert exit_code == 0
            mock_get_ids.assert_called_once_with(category='binary')
    finally:
        # Restore original sys.argv
        sys.argv = original_argv 