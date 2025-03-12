"""
Tests for the KnowledgeBase class.
"""
import os
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from zerotune import KnowledgeBase


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""
    
    def test_init(self):
        """Test initializing a new knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(name='test_kb', base_dir=tmpdir)
            
            # Check that the KB was initialized correctly
            assert kb.name == 'test_kb'
            assert Path(kb.kb_dir).exists()
            assert Path(kb.kb_dir).is_dir()
            
            # Check that the KB structure was created
            assert Path(os.path.join(kb.kb_dir, 'datasets')).exists()
            assert Path(os.path.join(kb.kb_dir, 'datasets')).is_dir()
            assert Path(os.path.join(kb.kb_dir, 'models')).exists()
            assert Path(os.path.join(kb.kb_dir, 'models')).is_dir()
            
            # Check that the knowledge base starts empty
            assert kb.kb is None
    
    def test_add_dataset(self):
        """Test adding a dataset to the knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(name='test_kb', base_dir=tmpdir)
            
            # Create a dataset
            X, y = make_classification(
                n_samples=100,
                n_features=5,
                n_informative=3,
                n_redundant=1,
                n_classes=2,
                random_state=42
            )
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            y_series = pd.Series(y, name='target')
            
            # Add the dataset
            kb.add_dataset(X_df, y_series, dataset_name='test_dataset')
            
            # Check that the dataset was added
            dataset_path = os.path.join(kb.kb_dir, 'datasets', 'test_dataset')
            assert Path(dataset_path).exists()
            assert Path(os.path.join(dataset_path, 'X.csv')).exists()
            assert Path(os.path.join(dataset_path, 'y.csv')).exists()
            assert Path(os.path.join(dataset_path, 'metadata.json')).exists()
    
    def test_add_synthetic_dataset(self):
        """Test adding a synthetic dataset to the knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(name='test_kb', base_dir=tmpdir)
            
            # Add a synthetic dataset
            kb.add_synthetic_dataset(n_samples=100, n_features=5, random_seed=42)
            
            # Since synthetic dataset names are auto-generated, we need to check if any dataset was created
            datasets_dir = os.path.join(kb.kb_dir, 'datasets')
            datasets = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
            
            assert len(datasets) > 0
            
            # Check the first dataset to ensure it's properly formed
            dataset_path = os.path.join(datasets_dir, datasets[0])
            assert Path(os.path.join(dataset_path, 'X.csv')).exists()
            assert Path(os.path.join(dataset_path, 'y.csv')).exists()
            assert Path(os.path.join(dataset_path, 'metadata.json')).exists()
    
    def test_compile_knowledge_base(self):
        """Test compiling the knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(name='test_kb', base_dir=tmpdir)
            
            # Add some datasets
            for i in range(3):
                X, y = make_classification(
                    n_samples=100,
                    n_features=5,
                    n_informative=3,
                    n_redundant=1,
                    n_classes=2,
                    random_state=i
                )
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                y_series = pd.Series(y, name='target')
                kb.add_dataset(X_df, y_series, dataset_name=f'test_dataset_{i}')
            
            # Define model and parameter config for evaluation
            model = DecisionTreeClassifier(random_state=42)
            param_config = {
                "max_depth": {
                    "percentage_splits": [0.25, 0.5, 0.7, 0.9],
                    "param_type": "int",
                    "dependency": "n_samples"
                },
                "min_samples_split": {
                    "percentage_splits": [0.01, 0.05, 0.1],
                    "param_type": "float"
                }
            }
            
            # Compile the knowledge base
            kb.compile_knowledge_base(
                model=model,
                param_config=param_config,
                n_random_configs=2,
                cv=2,
                metric='accuracy',
                random_seed=42
            )
            
            # Check that the knowledge base was compiled
            assert kb.kb is not None
            assert isinstance(kb.kb, pd.DataFrame)
            assert len(kb.kb) > 0
            
            # Check that all the expected columns are present
            expected_columns = [
                'dataset_name', 'n_samples', 'n_features', 'params_max_depth',
                'params_min_samples_split', 'performance'
            ]
            for col in expected_columns:
                assert col in kb.kb.columns
    
    def test_save_and_load(self):
        """Test saving and loading a knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and compile a KB
            kb = KnowledgeBase(name='test_kb', base_dir=tmpdir)
            
            # Add a dataset
            X, y = make_classification(
                n_samples=100,
                n_features=5,
                n_informative=3,
                n_redundant=1,
                n_classes=2,
                random_state=42
            )
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            y_series = pd.Series(y, name='target')
            kb.add_dataset(X_df, y_series, dataset_name='test_dataset')
            
            # Define model and parameter config for evaluation
            model = DecisionTreeClassifier(random_state=42)
            param_config = {
                "max_depth": {
                    "percentage_splits": [0.25, 0.9],
                    "param_type": "int",
                    "dependency": "n_samples"
                },
                "min_samples_split": {
                    "percentage_splits": [0.01, 0.1],
                    "param_type": "float"
                }
            }
            
            # Compile the knowledge base
            kb.compile_knowledge_base(
                model=model,
                param_config=param_config,
                n_random_configs=1,
                cv=2,
                metric='accuracy',
                random_seed=42
            )
            
            # Save the KB
            kb.save()
            
            # Load the KB
            kb2 = KnowledgeBase(name='test_kb', base_dir=tmpdir)
            kb2.load()
            
            # Check that the loaded KB is not None
            assert kb2.kb is not None
            assert isinstance(kb2.kb, pd.DataFrame)
            
            # Check that both dataframes have the same columns (possibly in different order)
            assert set(kb.kb.columns) == set(kb2.kb.columns)
            
            # Check that both dataframes have the same number of rows
            assert len(kb.kb) == len(kb2.kb)
            
            # Check that key columns have the same values
            for col in ['n_samples', 'n_features']:
                if col in kb.kb.columns and col in kb2.kb.columns:
                    assert kb.kb[col].equals(kb2.kb[col]), f"Column {col} values don't match"
    
    def test_train_model(self):
        """Test training a model from the knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and compile a KB
            kb = KnowledgeBase(name='test_kb', base_dir=tmpdir)
            
            # Add a few datasets
            for i in range(5):
                X, y = make_classification(
                    n_samples=100,
                    n_features=5,
                    n_informative=3,
                    n_redundant=1,
                    n_classes=2,
                    random_state=i
                )
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                y_series = pd.Series(y, name='target')
                kb.add_dataset(X_df, y_series, dataset_name=f'test_dataset_{i}')
            
            # Define model and parameter config for evaluation
            model = DecisionTreeClassifier(random_state=42)
            param_config = {
                "max_depth": {
                    "percentage_splits": [0.25, 0.5, 0.7, 0.9],
                    "param_type": "int",
                    "dependency": "n_samples"
                },
                "min_samples_split": {
                    "percentage_splits": [0.01, 0.05, 0.1],
                    "param_type": "float"
                }
            }
            
            # Compile the knowledge base
            kb.compile_knowledge_base(
                model=model,
                param_config=param_config,
                n_random_configs=2,
                cv=2,
                metric='accuracy',
                random_seed=42
            )
            
            # Define features and target parameters for training
            dataset_features = ['n_samples', 'n_features']
            target_params = ['params_max_depth', 'params_min_samples_split']
            
            # Train a model
            model, score = kb.train_model(
                dataset_features=dataset_features,
                target_params=target_params,
                random_seed=42
            )
            
            # Check that the model was trained
            assert model is not None
            assert score is not None
            
            # Check that the model was saved
            model_path = os.path.join(kb.kb_dir, 'models', 'zerotune_model.joblib')
            assert Path(model_path).exists() 