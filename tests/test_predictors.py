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
    
    def test_zerotune_predictor_predict(self, small_classification_dataset):
        """Test prediction with a ZeroTunePredictor."""
        X, y = small_classification_dataset
        predictor = ZeroTunePredictor(model_name='decision_tree')
        
        # We'll create a minimal mock implementation to test the functionality without relying on exact model behavior
        # Replace the predict method to avoid feature mismatch issues
        def mock_predict(features):
            # Return mock predictions
            return np.array([[10, 0.1, 0.05, 0.7]])
        
        # Temporarily replace the predict method
        original_predict = predictor.model.predict
        predictor.model.predict = mock_predict
        
        try:
            # Get predictions
            hyperparams = predictor.predict(X, y)
            
            # Check that predictions contain expected hyperparameters
            assert isinstance(hyperparams, dict)
            assert 'max_depth' in hyperparams
            assert 'min_samples_split' in hyperparams
            assert 'min_samples_leaf' in hyperparams
            assert 'max_features' in hyperparams
            
            # Check that hyperparameter values are reasonable
            assert isinstance(hyperparams['max_depth'], int)
            assert hyperparams['max_depth'] > 0
            assert isinstance(hyperparams['min_samples_split'], float)
            assert 0 < hyperparams['min_samples_split'] <= 1
            assert isinstance(hyperparams['min_samples_leaf'], float)
            assert 0 < hyperparams['min_samples_leaf'] <= 1
            assert isinstance(hyperparams['max_features'], float)
            assert 0 < hyperparams['max_features'] <= 1
        finally:
            # Restore the original predict method
            predictor.model.predict = original_predict


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