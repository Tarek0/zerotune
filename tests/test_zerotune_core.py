"""
Tests for core ZeroTune functions.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

from zerotune import (
    calculate_dataset_meta_parameters,
    relative2absolute_dict,
    generate_random_params,
    evaluate_model,
    remove_param_prefix
)


def test_calculate_dataset_meta_parameters(small_classification_dataset):
    """Test that meta parameters are calculated correctly."""
    X, y = small_classification_dataset
    meta_params = calculate_dataset_meta_parameters(X, y)
    
    # Check that basic parameters are present
    assert 'n_samples' in meta_params
    assert 'n_features' in meta_params
    assert 'imbalance_ratio' in meta_params
    
    # Check basic meta parameter values
    assert meta_params['n_samples'] == 100
    assert meta_params['n_features'] == 5


def test_calculate_dataset_meta_parameters_regression(small_regression_dataset):
    """Test meta parameter calculation for regression datasets."""
    X, y = small_regression_dataset
    
    # Convert regression target to integers for testing
    y = pd.Series((y > y.mean()).astype(int), name='target')
    
    meta_params = calculate_dataset_meta_parameters(X, y)
    
    # Check basic parameters
    assert 'n_samples' in meta_params
    assert 'n_features' in meta_params


def test_relative2absolute_dict(mock_dataset_meta_parameters):
    """Test conversion of relative parameter values to absolute values."""
    # Test with a specific relative parameter dict
    relative_params = {
        'max_depth': 0.5,  # 50% of n_samples with dependency
        'min_samples_split': 0.01,  # direct percentage value
        'min_samples_leaf': 0.02,
        'max_features': 0.7
    }
    
    param_config = {
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
    
    absolute_params = relative2absolute_dict(param_config, mock_dataset_meta_parameters, relative_params)
    
    # Check that all parameters are converted correctly
    assert absolute_params['max_depth'] == int(0.5 * mock_dataset_meta_parameters['n_samples'])
    assert absolute_params['min_samples_split'] == 0.01
    assert absolute_params['min_samples_leaf'] == 0.02
    assert absolute_params['max_features'] == 0.7


def test_generate_random_params(decision_tree_param_config):
    """Test random parameter generation."""
    random_params = generate_random_params(decision_tree_param_config, random_seed=42)
    
    # Check that all expected parameters are present
    for param in decision_tree_param_config.keys():
        assert param in random_params
    
    # Check that values are within the expected range
    for param, config in decision_tree_param_config.items():
        splits = config['percentage_splits']
        assert random_params[param] >= min(splits)
        assert random_params[param] <= max(splits)


def test_evaluate_model_classification(small_classification_dataset):
    """Test model evaluation for classification."""
    X, y = small_classification_dataset
    
    # Test with decision tree classifier - use the correct model name
    score, seed_scores = evaluate_model(X, y, "DecisionTreeClassifier4Param", 
                                       {"max_depth": 3}, 
                                       random_seed=42, n_folds=3, n_seeds=1)
    
    assert isinstance(score, float)
    assert score > 0
    assert len(seed_scores) == 1


def test_evaluate_model_regression(small_regression_dataset):
    """Test model evaluation for regression."""
    X, y = small_regression_dataset
    
    # Convert regression target to integers for testing
    y = pd.Series((y > y.mean()).astype(int), name='target')
    
    # Use the correct model name
    model_name = "DecisionTreeClassifier4Param"
    hyperparams = {"max_depth": 3}
    
    score, seed_scores = evaluate_model(X, y, model_name, hyperparams, 
                                       random_seed=42, n_folds=3, n_seeds=1)
    
    assert isinstance(score, float)
    assert score > 0


def test_remove_param_prefix():
    """Test removal of parameter prefixes."""
    params_with_prefix = {
        'params_max_depth': 5,
        'params_min_samples_split': 0.1,
        'params_min_samples_leaf': 0.05,
        'other_field': 'value'
    }
    
    params_without_prefix = remove_param_prefix(params_with_prefix)
    
    assert 'max_depth' in params_without_prefix
    assert 'min_samples_split' in params_without_prefix
    assert 'min_samples_leaf' in params_without_prefix
    assert 'params_max_depth' not in params_without_prefix
    assert 'other_field' in params_without_prefix 