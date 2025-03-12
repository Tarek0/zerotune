"""
Tests for the predictors module.
"""
import os
import pytest
import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification

from zerotune import (
    get_available_models,
    ZeroTunePredictor,
    CustomZeroTunePredictor
)


class TestPredictors:
    """Tests for the predictors module."""
    
    def test_get_available_models(self):
        """Test getting a list of available models."""
        models = get_available_models()
        
        # At least 'decision_tree' should be available
        assert isinstance(models, list)
        assert 'decision_tree' in models
    
    def test_zerotune_predictor_initialization(self):
        """Test initializing a ZeroTunePredictor."""
        # Initialize with a known model
        predictor = ZeroTunePredictor(model_name='decision_tree')
        
        # Check that the predictor was initialized correctly
        assert predictor.model_name == 'decision_tree'
        assert predictor.model is not None
        assert predictor.dataset_features is not None
        assert predictor.target_params is not None
        
        # Test with an unknown model name
        with pytest.raises(ValueError):
            ZeroTunePredictor(model_name='nonexistent_model')
    
    def test_zerotune_predictor_get_model_info(self):
        """Test getting model info from a ZeroTunePredictor."""
        predictor = ZeroTunePredictor(model_name='decision_tree')
        model_info = predictor.get_model_info()
        
        # Check that model info contains expected fields
        assert 'model_name' in model_info
        assert 'model_type' in model_info
        assert 'target_params' in model_info
        assert 'dataset_features' in model_info
        
        # Check specific values
        assert model_info['model_name'] == 'decision_tree'
        assert isinstance(model_info['target_params'], list)
        assert isinstance(model_info['dataset_features'], list)
    
    @pytest.mark.parametrize("model_name", ["decision_tree"])
    def test_zerotune_predictor_predict(self, small_classification_dataset, model_name):
        """
        Test prediction with a ZeroTunePredictor for various models.
        
        This parametrized test runs the same test logic with different model names,
        allowing for easier testing of multiple models.
        
        Args:
            small_classification_dataset: Fixture providing test data
            model_name: Name of the model to test
        """
        X, y = small_classification_dataset
        predictor = ZeroTunePredictor(model_name=model_name)
        
        # We'll create a minimal mock implementation to test the functionality without relying on exact model behavior
        # Replace the predict method to avoid feature mismatch issues
        def mock_predict(features):
            # Return mock predictions based on the model
            if model_name == 'decision_tree':
                return np.array([[10, 0.1, 0.05, 0.7]])
            else:
                # Default mock prediction with 4 parameters
                return np.array([[5, 0.2, 0.1, 0.5]])
        
        # Temporarily replace the predict method
        original_predict = predictor.model.predict
        predictor.model.predict = mock_predict
        
        try:
            # Get predictions
            hyperparams = predictor.predict(X, y)
            
            # Check that predictions are a dictionary
            assert isinstance(hyperparams, dict)
            
            # Check that at least some hyperparameters are present
            assert len(hyperparams) > 0
            
            # For decision tree, check specific parameters
            if model_name == 'decision_tree':
                assert 'max_depth' in hyperparams
                assert 'min_samples_split' in hyperparams
                assert isinstance(hyperparams['max_depth'], int)
                assert hyperparams['max_depth'] > 0
            
        finally:
            # Restore the original predict method
            predictor.model.predict = original_predict
    
    @pytest.mark.parametrize("dataset_features, target_params", [
        (["n_samples", "n_features"], ["params_max_depth", "params_min_samples_split"]),
        (["n_samples", "n_features", "imbalance_ratio"], ["params_max_depth", "params_min_samples_split", "params_min_samples_leaf"]),
    ])
    def test_custom_model_with_different_features(self, dataset_features, target_params, decision_tree_param_config, small_classification_dataset):
        """
        Test CustomZeroTunePredictor with different feature and target parameter sets.
        
        This parametrized test creates custom models with different combinations
        of dataset features and target parameters to verify flexibility.
        
        Args:
            dataset_features: List of features to use for prediction
            target_params: List of target parameters to predict
            decision_tree_param_config: Parameter configuration fixture
            small_classification_dataset: Test dataset fixture
        """
        # Create a temporary model for testing
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            # Create a simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            
            # Create synthetic training data matching the requested features/targets
            X_train = pd.DataFrame({
                feature: np.random.rand(5) * 100 for feature in dataset_features
            })
            
            y_train = pd.DataFrame({
                param: np.random.rand(5) for param in target_params
            })
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Create model data dict
            model_data = {
                'model': model,
                'dataset_features': dataset_features,
                'target_params': target_params,
                'score': 0.95
            }
            
            # Save the model
            joblib.dump(model_data, tmp.name)
            
            try:
                # Now test with this custom model
                X, y = small_classification_dataset
                predictor = CustomZeroTunePredictor(
                    model_path=tmp.name,
                    param_config=decision_tree_param_config
                )
                
                # Check that the model was loaded correctly
                assert predictor.dataset_features == dataset_features
                assert predictor.target_params == target_params
                
                # Mock the predict method to return suitable values
                def mock_predict(features):
                    # Return appropriate number of predictions
                    return np.array([[0.5] * len(target_params)])
                
                # Temporarily replace the predict method
                original_predict = predictor.model.predict
                predictor.model.predict = mock_predict
                
                try:
                    # Test prediction
                    hyperparams = predictor.predict(X, y)
                    
                    # Check that all expected target params are predicted
                    # Strip 'params_' prefix for comparison with hyperparams keys
                    expected_params = [p.replace('params_', '') for p in target_params]
                    for param in expected_params:
                        assert param in hyperparams
                    
                finally:
                    # Restore original predict method
                    predictor.model.predict = original_predict
            
            finally:
                # Clean up the temporary file
                os.unlink(tmp.name)


class TestCustomZeroTunePredictor:
    """Tests for the CustomZeroTunePredictor class."""
    
    @pytest.fixture
    def custom_model_path(self):
        """Fixture that creates a custom model for testing."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            # Create a simple RandomForestRegressor model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            
            # Create some training data and fit the model
            X, y = make_classification(
                n_samples=100,
                n_features=3,
                n_informative=2,
                n_redundant=0,
                n_classes=2,
                random_state=42
            )
            
            # Extract meta features that will be dataset_features
            dataset_features = ['n_samples', 'n_features']
            
            # Define target parameters we're trying to predict
            target_params = ['params_max_depth', 'params_min_samples_split']
            
            # Create synthetic training data for the model
            train_X = pd.DataFrame({
                'n_samples': [100, 200, 300, 400, 500],
                'n_features': [5, 10, 15, 20, 25],
            })
            
            train_y = pd.DataFrame({
                'params_max_depth': [5, 10, 15, 20, 25],
                'params_min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
            
            # Fit the model
            model.fit(train_X, train_y)
            
            # Create model data dict
            model_data = {
                'model': model,
                'dataset_features': dataset_features,
                'target_params': target_params,
                'score': 0.95
            }
            
            # Save the model
            joblib.dump(model_data, tmp.name)
            
            yield tmp.name
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_custom_zerotune_predictor_initialization(self, custom_model_path, decision_tree_param_config):
        """Test initializing a CustomZeroTunePredictor."""
        predictor = CustomZeroTunePredictor(
            model_path=custom_model_path,
            param_config=decision_tree_param_config
        )
        
        # Check that the predictor was initialized correctly
        assert predictor.model is not None
        assert predictor.dataset_features is not None
        assert predictor.target_params is not None
        assert predictor.param_config == decision_tree_param_config
    
    def test_custom_zerotune_predictor_predict(self, custom_model_path, decision_tree_param_config, small_classification_dataset):
        """Test prediction with a CustomZeroTunePredictor."""
        X, y = small_classification_dataset
        predictor = CustomZeroTunePredictor(
            model_path=custom_model_path,
            param_config=decision_tree_param_config
        )
        
        # We'll create a minimal mock implementation to test the functionality without relying on exact model behavior
        # Replace the predict method to avoid feature mismatch issues
        def mock_predict(features):
            # Return mock predictions
            return np.array([[10, 0.1]])
        
        # Temporarily replace the predict method
        original_predict = predictor.model.predict
        predictor.model.predict = mock_predict
        
        try:
            # Get predictions
            hyperparams = predictor.predict(X, y)
            
            # Check that predictions contain expected hyperparameters
            assert isinstance(hyperparams, dict)
            
            # The prediction might not contain all hyperparameters from param_config
            # only those present in the model's target_params
            model_data = joblib.load(custom_model_path)
            target_params = [p.replace('params_', '') for p in model_data['target_params']]
            
            for param in target_params:
                assert param in hyperparams
        finally:
            # Restore the original predict method
            predictor.model.predict = original_predict 