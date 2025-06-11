"""
Tests for save_knowledge_base and load_knowledge_base functions.
"""

import pytest
import os
import tempfile
from zerotune.core.knowledge_base import (
    initialize_knowledge_base,
    save_knowledge_base,
    load_knowledge_base
)


def test_save_and_load_knowledge_base():
    """Test saving and loading a knowledge base."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = os.path.join(temp_dir, "test_kb.json")
        
        # Create a knowledge base with some data
        kb = initialize_knowledge_base()
        
        # Add meta features
        kb["meta_features"] = [
            {"n_samples": 100, "n_features": 10},
            {"n_samples": 200, "n_features": 20}
        ]
        
        # Add results
        kb["results"] = [
            {
                "dataset_name": "test_dataset_1",
                "best_hyperparameters": {"max_depth": 5},
                "best_score": 0.85,
                "model_type": "decision_tree"
            },
            {
                "dataset_name": "test_dataset_2",
                "best_hyperparameters": {"n_estimators": 100},
                "best_score": 0.90,
                "model_type": "random_forest"
            }
        ]
        
        # Add datasets
        kb["datasets"] = [
            {"name": "test_dataset_1", "id": 123},
            {"name": "test_dataset_2", "id": 456}
        ]
        
        # Save the knowledge base
        success = save_knowledge_base(kb, kb_path)
        assert success
        
        # Verify the file exists
        assert os.path.exists(kb_path)
        
        # Load the knowledge base
        loaded_kb = load_knowledge_base(kb_path)
        
        # Verify the loaded KB has the correct structure
        assert "meta_features" in loaded_kb
        assert "results" in loaded_kb
        assert "datasets" in loaded_kb
        assert "created_at" in loaded_kb
        assert "updated_at" in loaded_kb
        
        # Verify the loaded data
        assert len(loaded_kb["meta_features"]) == 2
        assert len(loaded_kb["results"]) == 2
        assert len(loaded_kb["datasets"]) == 2
        
        # Verify the first meta feature
        assert loaded_kb["meta_features"][0]["n_samples"] == 100
        assert loaded_kb["meta_features"][0]["n_features"] == 10
        
        # Verify the first result
        assert loaded_kb["results"][0]["dataset_name"] == "test_dataset_1"
        assert loaded_kb["results"][0]["best_hyperparameters"]["max_depth"] == 5
        assert loaded_kb["results"][0]["best_score"] == 0.85
        
        # Test loading a non-existent file (should return an empty knowledge base)
        non_existent_path = os.path.join(temp_dir, "nonexistent.json")
        empty_kb = load_knowledge_base(non_existent_path)
        assert "meta_features" in empty_kb
        assert len(empty_kb["meta_features"]) == 0
        assert os.path.exists(non_existent_path)  # Should create the file


def test_save_and_load_with_model_type():
    """Test saving and loading a knowledge base with model type filter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = os.path.join(temp_dir, "test_kb.json")
        
        # Create a knowledge base with some data
        kb = initialize_knowledge_base()
        
        # Add meta features
        kb["meta_features"] = [
            {"n_samples": 100, "n_features": 10}
        ]
        
        # Add results for multiple model types
        kb["results"] = [
            {
                "dataset_name": "test_dataset",
                "best_hyperparameters": {"max_depth": 5},
                "best_score": 0.85,
                "model_type": "decision_tree"
            },
            {
                "dataset_name": "test_dataset",
                "best_hyperparameters": {"n_estimators": 100},
                "best_score": 0.90,
                "model_type": "random_forest"
            }
        ]
        
        # Add datasets
        kb["datasets"] = [
            {"name": "test_dataset", "id": 123}
        ]
        
        # Save the knowledge base
        save_knowledge_base(kb, kb_path)
        
        # Load the knowledge base with model type filter
        dt_kb = load_knowledge_base(kb_path, model_type="decision_tree")
        
        # Verify the loaded KB only has decision tree results
        assert len(dt_kb["results"]) == 1
        assert dt_kb["results"][0]["model_type"] == "decision_tree" 