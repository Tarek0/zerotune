"""
Tests for the model configurations module in ZeroTune.
"""

import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from zerotune.core.model_configs import ModelConfigs


def test_model_configs_structure():
    """Test that all model configurations have the expected structure."""
    # Test all three model types to ensure they follow the same pattern
    configs = [
        ('decision_tree', ModelConfigs.get_decision_tree_config(), DecisionTreeClassifier),
        ('random_forest', ModelConfigs.get_random_forest_config(), RandomForestClassifier),
        ('xgboost', ModelConfigs.get_xgboost_config(), XGBClassifier)
    ]
    
    for model_name, config, expected_class in configs:
        # Verify the config has the expected keys
        assert "name" in config
        assert "model" in config
        assert "metric" in config
        assert "param_config" in config
        
        # Verify the model instance
        assert isinstance(config["model"], expected_class)
        assert hasattr(config["model"], "random_state")
        assert config["model"].random_state == 42
        
        # Verify param_config structure
        param_config = config["param_config"]
        assert len(param_config) > 0
        
        for param, param_settings in param_config.items():
            assert "param_type" in param_settings
            # Verify each parameter has either percentage_splits or min/max values
            has_percentage_splits = "percentage_splits" in param_settings
            has_min_max = "min_value" in param_settings and "max_value" in param_settings
            assert has_percentage_splits or has_min_max, f"Parameter {param} in {model_name} config missing range definition" 