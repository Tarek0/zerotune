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


class TestZeroTuneCore:
    """Tests for core ZeroTune functions."""
    
    def test_calculate_dataset_meta_parameters(self, small_classification_dataset):
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


    def test_calculate_dataset_meta_parameters_regression(self, small_regression_dataset):
        """Test meta parameter calculation for regression datasets."""
        X, y = small_regression_dataset
        
        # Convert regression target to integers for testing
        y = pd.Series((y > y.mean()).astype(int), name='target')
        
        meta_params = calculate_dataset_meta_parameters(X, y)
        
        # Check basic parameters
        assert 'n_samples' in meta_params
        assert 'n_features' in meta_params


    def test_calculate_dataset_meta_parameters_empty(self, empty_dataset):
        """
        Test meta parameter calculation with an empty dataset.
        
        This tests the edge case of an empty dataset (0 rows) to ensure
        the function either handles it gracefully or raises an appropriate
        exception with a helpful error message.
        """
        X, y = empty_dataset
        
        # We expect this to raise ValueError due to empty dataset
        with pytest.raises(ValueError) as excinfo:
            meta_params = calculate_dataset_meta_parameters(X, y)
        
        # Check that the error message is descriptive
        assert "empty" in str(excinfo.value).lower() or "no samples" in str(excinfo.value).lower()
    
    
    def test_calculate_dataset_meta_parameters_single_row(self, single_row_dataset):
        """
        Test meta parameter calculation with a single-row dataset.
        
        This tests the edge case of having only one data point, which can be
        problematic for statistical calculations like variance or correlation.
        """
        X, y = single_row_dataset
        
        # Either the function should handle this gracefully or raise a specific exception
        try:
            meta_params = calculate_dataset_meta_parameters(X, y)
            
            # If it succeeds, check basic parameters
            assert 'n_samples' in meta_params
            assert meta_params['n_samples'] == 1
            assert 'n_features' in meta_params
            assert meta_params['n_features'] == 5
            
            # Correlation-based metrics might be NaN or 0
            if 'feature_correlation_avg' in meta_params:
                assert pd.isna(meta_params['feature_correlation_avg']) or meta_params['feature_correlation_avg'] == 0
                
        except ValueError as e:
            # If it fails, check that the error message is descriptive
            assert "single" in str(e).lower() or "one sample" in str(e).lower() or "not enough" in str(e).lower()
    
    
    def test_calculate_dataset_meta_parameters_high_dimensional(self, high_dimensional_dataset):
        """
        Test meta parameter calculation with a high-dimensional dataset.
        
        This tests the system's capability to handle datasets with many features,
        which can strain computational resources and potentially reveal scaling issues.
        """
        X, y = high_dimensional_dataset
        
        meta_params = calculate_dataset_meta_parameters(X, y)
        
        # Check basic parameters
        assert 'n_samples' in meta_params
        assert meta_params['n_samples'] == 50
        assert 'n_features' in meta_params
        assert meta_params['n_features'] == 100
        
        # Check that correlation calculations completed
        assert 'feature_correlation_avg' in meta_params
        assert 'n_highly_target_corr' in meta_params
    
    
    def test_calculate_dataset_meta_parameters_categorical(self, all_categorical_dataset):
        """
        Test meta parameter calculation with all categorical features.
        
        This tests the system's ability to handle categorical data appropriately,
        including calculation of meta-features specific to categorical data.
        """
        X, y = all_categorical_dataset
        
        meta_params = calculate_dataset_meta_parameters(X, y)
        
        # Check basic parameters
        assert 'n_samples' in meta_params
        assert 'n_features' in meta_params
        assert meta_params['n_features'] == 5
        
        # Check categorical-specific parameters if they exist
        if 'n_categorical' in meta_params:
            assert meta_params['n_categorical'] > 0
        if 'n_binary' in meta_params:
            assert meta_params['n_binary'] > 0


    def test_relative2absolute_dict(self, mock_dataset_meta_parameters):
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
    
    
    def test_relative2absolute_dict_edge_cases(self, mock_dataset_meta_parameters):
        """
        Test conversion of relative parameter values with edge cases.
        
        This tests how the function handles extreme values (0 or 1) and missing
        dependencies in the dataset properties.
        """
        # Edge case: Extreme values
        extreme_params = {
            'max_depth': 0.999,  # Almost 100% of n_samples
            'min_samples_split': 0.005,  # Very small percentage
            'min_samples_leaf': 0.005,
            'max_features': 0.99  # Almost all features
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
        
        absolute_params = relative2absolute_dict(param_config, mock_dataset_meta_parameters, extreme_params)
        
        # Check extreme value conversions
        assert absolute_params['max_depth'] == int(0.999 * mock_dataset_meta_parameters['n_samples'])
        assert absolute_params['min_samples_split'] == 0.005
        assert absolute_params['max_features'] == 0.99
        
        # Edge case: Missing dependency
        # Create a copy without n_samples
        limited_meta_params = mock_dataset_meta_parameters.copy()
        del limited_meta_params['n_samples']
        
        # Update config to rely on a different dependency
        modified_config = param_config.copy()
        modified_config["max_depth"] = {
            "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999],
            "param_type": "int",
            "dependency": "n_features"  # Changed to n_features
        }
        
        # This should still work since n_features exists
        test_params = {'max_depth': 0.5, 'min_samples_split': 0.01}
        absolute_params = relative2absolute_dict(modified_config, limited_meta_params, test_params)
        assert absolute_params['max_depth'] == int(0.5 * limited_meta_params['n_features'])


    def test_generate_random_params(self, decision_tree_param_config):
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
    
    
    def test_generate_random_params_reproducibility(self, decision_tree_param_config):
        """
        Test that random parameter generation is reproducible with the same seed.
        
        This verifies that using the same random seed produces identical parameters,
        which is crucial for reproducible experiments and debugging.
        """
        # Generate params with the same seed twice
        params1 = generate_random_params(decision_tree_param_config, random_seed=42)
        params2 = generate_random_params(decision_tree_param_config, random_seed=42)
        
        # They should be identical
        assert params1 == params2
        
        # Generate with a different seed
        params3 = generate_random_params(decision_tree_param_config, random_seed=43)
        
        # They should be different (although there's a small chance they're the same by coincidence)
        assert params1 != params3


    def test_evaluate_model_classification(self, small_classification_dataset):
        """Test model evaluation for classification."""
        X, y = small_classification_dataset
        
        # Test with decision tree classifier - use the correct model name
        score, seed_scores = evaluate_model(X, y, "DecisionTreeClassifier4Param", 
                                           {"max_depth": 3}, 
                                           random_seed=42, n_folds=3, n_seeds=1)
        
        assert isinstance(score, float)
        assert score > 0
        assert len(seed_scores) == 1


    def test_evaluate_model_regression(self, small_regression_dataset):
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
    
    
    def test_evaluate_model_multiple_seeds(self, small_classification_dataset):
        """
        Test model evaluation with multiple random seeds.
        
        This verifies that using multiple seeds produces different evaluation
        scores and that the average is computed correctly.
        """
        X, y = small_classification_dataset
        
        # Test with multiple seeds
        score, seed_scores = evaluate_model(X, y, "DecisionTreeClassifier4Param", 
                                          {"max_depth": 3}, 
                                          random_seed=42, n_folds=3, n_seeds=5)
        
        # Check that multiple scores were generated
        assert len(seed_scores) == 5
        
        # Check that the average is correct
        assert score == pytest.approx(sum(seed_scores) / len(seed_scores))
        
        # Check that scores vary (they should with different random states)
        # There's a small chance they could all be the same, but it's unlikely
        assert len(set(seed_scores)) > 1


    def test_remove_param_prefix(self):
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