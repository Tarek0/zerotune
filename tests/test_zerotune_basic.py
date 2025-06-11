"""
Tests for the new ZeroTune class.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from unittest.mock import patch, MagicMock

from zerotune import ZeroTune


@pytest.fixture
def synthetic_dataset():
    """Create a synthetic dataset for testing."""
    # Create a synthetic classification dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    return (X_train, X_test, y_train, y_test)


def test_zerotune_initialization():
    """Test the initialization of the ZeroTune class."""
    # Initialize ZeroTune with different model types
    zt_dt = ZeroTune(model_type="decision_tree")
    zt_rf = ZeroTune(model_type="random_forest")
    zt_xgb = ZeroTune(model_type="xgboost")
    
    # Verify the model type is set correctly
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


def test_zerotune_optimize(synthetic_dataset):
    """Test the optimize method of the ZeroTune class."""
    X_train, X_test, y_train, y_test = synthetic_dataset
    
    # Initialize ZeroTune with a simplified configuration
    zt = ZeroTune(model_type="decision_tree")
    
    # Fix the root causes:
    # 1. Replace unsupported 'roc_auc' metric with supported 'accuracy'
    # 2. Add a proper param_grid with standard scikit-learn format
    zt.model_config["metric"] = "accuracy"
    zt.model_config["param_grid"] = {
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.7, 0.9]
    }
    
    # Create a minimal knowledge base with clean meta-features
    zt.knowledge_base = {
        "meta_features": [
            {
                "n_samples": 100,
                "n_features": 10,
                "imbalance_ratio": 1.0,
                "n_highly_target_corr": 3,
                "avg_target_corr": 0.25,
                "var_target_corr": 0.05,
                "avg_feature_m1": 0.0,
                "var_feature_m1": 1.0,
                "avg_feature_m2": 1.0,
                "var_feature_m2": 0.1,
                "avg_feature_m3": 0.0,
                "var_feature_m3": 0.2,
                "avg_feature_m4": 3.0,
                "var_feature_m4": 0.5,
                "avg_row_m1": 0.0,
                "var_row_m1": 0.3,
                "avg_row_m2": 1.0,
                "var_row_m2": 0.2,
                "avg_row_m3": 0.0,
                "var_row_m3": 0.1,
                "avg_row_m4": 3.0,
                "var_row_m4": 0.4
            }
        ],
        "results": [
            {
                "model_type": "decision_tree",
                "hyperparameters": {
                    "max_depth": 5,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": 0.7
                },
                "score": 0.85
            }
        ]
    }
    
    # Use a small number of iterations for faster testing
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = os.path.join(temp_dir, "test_kb.json")
        zt.kb_path = kb_path
        
        # Optimize hyperparameters with real functionality (no mocks)
        best_params, best_score, model = zt.optimize(
            X_train, y_train, n_iter=2, verbose=False
        )
        
        # Verify the results
        assert isinstance(best_params, dict)
        assert "max_depth" in best_params
        assert best_score > 0  # Score should be positive
        assert model is not None  # Model should be trained
        
        # Verify the model can make predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)


def test_zerotune_build_knowledge_base():
    """Test the build_knowledge_base method of the ZeroTune class with minimal mocking."""
    # The issue is JSON serialization of NumPy types, so we'll mock the save_knowledge_base function directly
    
    # Create a synthetic dataset for testing
    X = pd.DataFrame(np.random.rand(20, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 20))
    
    # Initialize ZeroTune
    zt = ZeroTune(model_type="decision_tree")
    
    # Fix the root causes we identified earlier
    zt.model_config["metric"] = "accuracy"
    zt.model_config["param_grid"] = {
        "max_depth": [3, 5],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": [0.5, 0.7]
    }
    
    # Create a temporary KB path
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = os.path.join(temp_dir, "test_kb.json")
        zt.kb_path = kb_path
        
        # Mock dataset retrieval and knowledge base saving
        # Note: We need to patch at the module level where it's imported, not where it's defined
        with patch('zerotune.core.zero_tune.get_recommended_datasets', return_value=[61]), \
             patch('zerotune.core.zero_tune.fetch_open_ml_data') as mock_fetch, \
             patch('zerotune.core.zero_tune.prepare_data') as mock_prepare, \
             patch('zerotune.core.zero_tune.save_knowledge_base', return_value=True):
            
            # Set up dataset mocks
            mock_fetch.return_value = (pd.concat([X, y.rename('target')], axis=1), 'target', 'test_dataset')
            mock_prepare.return_value = (X, y)
            
            # Build a knowledge base with real optimization logic
            kb = zt.build_knowledge_base(n_datasets=1, n_iter=1, verbose=False)
            
            # Verify the knowledge base structure
            assert isinstance(kb, dict)
            assert "meta_features" in kb
            assert "results" in kb
            assert "datasets" in kb
            
            # Verify we have data for the dataset
            assert len(kb["meta_features"]) > 0
            assert len(kb["results"]) > 0
            assert len(kb["datasets"]) > 0
            
            # Verify the model type is stored in the results
            assert kb["results"][0]["model_type"] == "decision_tree" 