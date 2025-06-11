"""
Tests for the model configurations module in ZeroTune.
"""

import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from zerotune.core.model_configs import ModelConfigs


def test_decision_tree_config():
    """Test decision tree model configuration."""
    # Get the configuration
    config = ModelConfigs.get_decision_tree_config()
    
    # Verify the config has the expected keys
    assert "name" in config
    assert "model" in config
    assert "metric" in config
    assert "param_config" in config
    
    # Verify the model class
    assert isinstance(config["model"], DecisionTreeClassifier)
    
    # Verify the metric
    assert config["metric"] == "roc_auc"
    
    # Verify parameters in param_config
    param_config = config["param_config"]
    assert "max_depth" in param_config
    assert "min_samples_split" in param_config
    assert "min_samples_leaf" in param_config
    assert "max_features" in param_config
    
    # Verify that all parameter configurations have required keys
    for param, param_settings in param_config.items():
        assert "percentage_splits" in param_settings
        assert "param_type" in param_settings


def test_random_forest_config():
    """Test random forest model configuration."""
    # Get the configuration
    config = ModelConfigs.get_random_forest_config()
    
    # Verify the config has the expected keys
    assert "name" in config
    assert "model" in config
    assert "metric" in config
    assert "param_config" in config
    
    # Verify the model class
    assert isinstance(config["model"], RandomForestClassifier)
    
    # Verify the metric
    assert config["metric"] == "roc_auc"
    
    # Verify parameters in param_config
    param_config = config["param_config"]
    assert "n_estimators" in param_config
    assert "max_depth" in param_config
    assert "min_samples_split" in param_config
    assert "min_samples_leaf" in param_config
    assert "max_features" in param_config
    
    # Verify that all parameter configurations have required keys
    for param, param_settings in param_config.items():
        assert "percentage_splits" in param_settings
        assert "param_type" in param_settings


def test_xgboost_config():
    """Test XGBoost model configuration."""
    # Get the configuration
    config = ModelConfigs.get_xgboost_config()
    
    # Verify the config has the expected keys
    assert "name" in config
    assert "model" in config
    assert "metric" in config
    assert "param_config" in config
    
    # Verify the model class
    assert isinstance(config["model"], XGBClassifier)
    
    # Verify the metric
    assert config["metric"] == "roc_auc"
    
    # Verify parameters in param_config
    param_config = config["param_config"]
    assert "n_estimators" in param_config
    assert "learning_rate" in param_config
    assert "max_depth" in param_config
    assert "subsample" in param_config
    assert "colsample_bytree" in param_config
    
    # Verify that all parameter configurations have required keys
    for param, param_settings in param_config.items():
        assert "percentage_splits" in param_settings
        assert "param_type" in param_settings


def test_all_configs_have_random_state():
    """Test that all model configurations include a random_state parameter."""
    # Get all model configurations
    dt_config = ModelConfigs.get_decision_tree_config()
    rf_config = ModelConfigs.get_random_forest_config()
    xgb_config = ModelConfigs.get_xgboost_config()
    
    # Verify all configs have a model with random_state
    assert hasattr(dt_config["model"], "random_state")
    assert hasattr(rf_config["model"], "random_state")
    assert hasattr(xgb_config["model"], "random_state")
    
    # Verify the random_state value is consistent
    assert dt_config["model"].random_state == 42
    assert rf_config["model"].random_state == 42
    assert xgb_config["model"].random_state == 42


def test_create_model_instance():
    """Test creating model instances from configurations."""
    # Get all model configurations
    dt_config = ModelConfigs.get_decision_tree_config()
    rf_config = ModelConfigs.get_random_forest_config()
    xgb_config = ModelConfigs.get_xgboost_config()
    
    # Verify model classes
    assert isinstance(dt_config["model"], DecisionTreeClassifier)
    assert isinstance(rf_config["model"], RandomForestClassifier)
    assert isinstance(xgb_config["model"], XGBClassifier) 