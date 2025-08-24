"""
Tests for the ZeroTune class core functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from unittest.mock import patch

from zerotune import ZeroTune


def test_zerotune_initialization():
    """Test the initialization of the ZeroTune class."""
    # Test valid model types
    zt_dt = ZeroTune(model_type="decision_tree")
    zt_rf = ZeroTune(model_type="random_forest")
    zt_xgb = ZeroTune(model_type="xgboost")
    
    assert zt_dt.model_type == "decision_tree"
    assert zt_rf.model_type == "random_forest"
    assert zt_xgb.model_type == "xgboost"
    
    # Verify the model configs were loaded correctly
    assert zt_dt.model_config["name"] == "DecisionTreeClassifier"
    assert zt_rf.model_config["name"] == "RandomForestClassifier"
    assert zt_xgb.model_config["name"] == "XGBClassifier"
    
    # Test with invalid model type
    with pytest.raises(ValueError):
        ZeroTune(model_type="invalid_model")


def test_zerotune_optimization_integration():
    """Test ZeroTune optimization integration with the core optimization module."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    zt = ZeroTune(model_type='decision_tree')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = os.path.join(temp_dir, "test_kb.json")
        zt.kb_path = kb_path
        
        # Test the core optimization workflow
        from zerotune.core.optimization import optimize_hyperparameters
        from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
        
        param_grid = zt._convert_param_config_to_grid(zt.model_config["param_config"], X_train.shape)
        meta_features = calculate_dataset_meta_parameters(X_train, y_train)
        
        best_params, best_score, all_results, df_trials = optimize_hyperparameters(
            zt.model_class,
            param_grid,
            X_train, 
            y_train,
            metric=zt.model_config["metric"],
            n_iter=2,
            verbose=False,
            dataset_meta_params=meta_features
        )
        
        # Verify the results are valid
        assert isinstance(best_params, dict)
        assert "max_depth" in best_params
        assert best_score > 0
        assert len(all_results) > 0
        
        # Test that we can create and use a model with the best params
        model = zt.model_class(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)


def test_zerotune_knowledge_base_building():
    """Test knowledge base building functionality."""
    zt = ZeroTune(model_type="decision_tree")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = os.path.join(temp_dir, "test_kb.json")
        zt.kb_path = kb_path
        
        # Mock external dependencies to focus on the knowledge base logic
        with patch('zerotune.core.zero_tune.get_recommended_datasets', return_value=[61]), \
             patch('zerotune.core.zero_tune.fetch_open_ml_data') as mock_fetch, \
             patch('zerotune.core.zero_tune.prepare_data') as mock_prepare, \
             patch('zerotune.core.zero_tune.save_knowledge_base', return_value=True):
            
            # Set up mocks with realistic data
            X = pd.DataFrame(np.random.rand(20, 5), columns=[f"feature_{i}" for i in range(5)])
            y = pd.Series(np.random.randint(0, 2, 20))
            
            mock_fetch.return_value = (pd.concat([X, y.rename('target')], axis=1), 'target', 'test_dataset')
            mock_prepare.return_value = (X, y)
            
            # Build knowledge base
            kb = zt.build_knowledge_base(n_datasets=1, n_iter=1, verbose=False)
            
            # Verify structure
            assert isinstance(kb, dict)
            assert "meta_features" in kb
            assert "results" in kb
            assert "datasets" in kb
            assert len(kb["results"]) > 0
            assert kb["results"][0]["model_type"] == "decision_tree" 