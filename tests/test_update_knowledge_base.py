"""
Tests for update_knowledge_base function.
"""

import pytest
from zerotune.core.knowledge_base import (
    initialize_knowledge_base,
    update_knowledge_base
)


@pytest.fixture
def synthetic_meta_parameters():
    """Create synthetic meta-parameters for testing."""
    return {
        "n_samples": 100,
        "n_features": 10,
        "imbalance_ratio": 1.0,
        "n_highly_target_corr": 3,
        "avg_target_corr": 0.25,
        "var_target_corr": 0.05,
        "avg_feature_m1": 0.0,
        "var_feature_m1": 1.0
    }


def test_update_knowledge_base():
    """Test updating a knowledge base with results from a new dataset."""
    # Create an empty knowledge base
    kb = initialize_knowledge_base()
    
    # Dataset information
    dataset_name = "test_dataset_1"
    dataset_id = 123
    meta_features = {
        "n_samples": 100,
        "n_features": 10,
        "imbalance_ratio": 1.0
    }
    
    # First model type (decision tree)
    dt_hyperparameters = {"max_depth": 5, "min_samples_split": 2}
    dt_score = 0.85
    
    # Update the knowledge base with decision tree results
    updated_kb = update_knowledge_base(
        kb, 
        dataset_name=dataset_name,
        meta_features=meta_features,
        best_hyperparameters=dt_hyperparameters,
        best_score=dt_score,
        dataset_id=dataset_id,
        model_type="decision_tree"
    )
    
    # Verify the knowledge base structure
    assert "meta_features" in updated_kb
    assert "results" in updated_kb
    assert "datasets" in updated_kb
    
    # Verify the meta features were added
    assert len(updated_kb["meta_features"]) == 1
    assert updated_kb["meta_features"][0] == meta_features
    
    # Verify the results were added
    assert len(updated_kb["results"]) == 1
    result = updated_kb["results"][0]
    assert result["dataset_name"] == dataset_name
    assert result["best_hyperparameters"] == dt_hyperparameters
    assert result["best_score"] == dt_score
    assert result["model_type"] == "decision_tree"
    assert result["dataset_id"] == dataset_id
    
    # Verify the dataset info was added
    assert len(updated_kb["datasets"]) == 1
    assert updated_kb["datasets"][0]["name"] == dataset_name
    assert updated_kb["datasets"][0]["id"] == dataset_id
    
    # Add a second model type (random forest)
    rf_hyperparameters = {"n_estimators": 100, "max_depth": 10}
    rf_score = 0.90
    
    # Update the knowledge base with random forest results
    updated_kb = update_knowledge_base(
        updated_kb,
        dataset_name=dataset_name,
        meta_features=meta_features,  # Same meta-features
        best_hyperparameters=rf_hyperparameters,
        best_score=rf_score,
        dataset_id=dataset_id,
        model_type="random_forest"
    )
    
    # Verify the meta features were not duplicated
    assert len(updated_kb["meta_features"]) == 1
    
    # Verify the new result was added
    assert len(updated_kb["results"]) == 2
    rf_result = [r for r in updated_kb["results"] if r["model_type"] == "random_forest"][0]
    assert rf_result["best_hyperparameters"] == rf_hyperparameters
    assert rf_result["best_score"] == rf_score
    
    # Verify the dataset info was not duplicated
    assert len(updated_kb["datasets"]) == 1 