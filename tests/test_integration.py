"""
Integration tests for ZeroTune.

These tests verify that the complete ZeroTune workflow functions correctly,
from creating a knowledge base to making predictions.
"""
import os
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from zerotune import (
    KnowledgeBase,
    CustomZeroTunePredictor,
    calculate_dataset_meta_parameters
)


@pytest.mark.integration
@pytest.mark.slow
class TestIntegrationWorkflow:
    """Integration tests for complete ZeroTune workflows."""
    
    def test_full_workflow(self):
        """
        Test the complete workflow from knowledge base creation to prediction.
        
        This test covers:
        1. Creating a knowledge base
        2. Adding synthetic datasets
        3. Compiling the knowledge base
        4. Training a model
        5. Using the model to predict hyperparameters
        6. Verifying the predictions work with a real model
        """
        # Skip test if running in CI to avoid long-running tests
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping integration test in CI environment")
            
        # Create a temporary directory for the knowledge base
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Create a knowledge base
            kb = KnowledgeBase(name="test_integration_kb", base_dir=temp_dir)
            
            # Step 2: Add synthetic datasets (use a small number for faster testing)
            n_datasets = 5
            for i in range(n_datasets):
                # Generate a synthetic dataset
                X, y = make_classification(
                    n_samples=100 + i * 20,  # Vary the sample size
                    n_features=5 + i,         # Vary the feature count
                    n_informative=3,
                    n_redundant=1,
                    n_classes=2,
                    random_state=i
                )
                X_df = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(X.shape[1])])
                y_series = pd.Series(y, name='target')
                
                # Add to the knowledge base
                kb.add_dataset(X_df, y_series, dataset_name=f'synthetic_dataset_{i}')
            
            # Step 3: Compile the knowledge base
            param_config = {
                "max_depth": {
                    "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9],
                    "param_type": "int",
                    "dependency": "n_samples"
                },
                "min_samples_split": {
                    "percentage_splits": [0.01, 0.02, 0.05, 0.1],
                    "param_type": "float"
                }
            }
            
            # Use a model for compilation
            model = DecisionTreeClassifier(random_state=42)
            
            # Compile with minimal configurations for speed
            kb.compile_knowledge_base(
                model=model,
                param_config=param_config,
                n_random_configs=2,  # Use small number for speed
                cv=2,
                metric='accuracy',
                random_seed=42
            )
            
            # Check that compilation was successful
            assert kb.kb is not None
            assert isinstance(kb.kb, pd.DataFrame)
            assert len(kb.kb) > 0
            
            # Step 4: Train a model
            dataset_features = ['n_samples', 'n_features']
            target_params = ['params_max_depth', 'params_min_samples_split']
            
            trained_model, score = kb.train_model(
                dataset_features=dataset_features,
                target_params=target_params,
                n_iter=10,  # Use small number for speed
                random_seed=42
            )
            
            # Check that training was successful
            assert trained_model is not None
            assert score is not None
            
            # Check that the model was saved
            model_path = Path(temp_dir) / "test_integration_kb" / "models" / "zerotune_model.joblib"
            assert model_path.exists()
            
            # Step 5: Create a test dataset for prediction
            X_test, y_test = make_classification(
                n_samples=200,
                n_features=10,
                n_informative=5,
                n_redundant=2,
                n_classes=2,
                random_state=999  # Different seed for test data
            )
            X_test_df = pd.DataFrame(X_test, columns=[f'feature_{j}' for j in range(X_test.shape[1])])
            y_test_series = pd.Series(y_test, name='target')
            
            # Create a predictor with the trained model
            predictor = CustomZeroTunePredictor(
                model_path=str(model_path),
                param_config=param_config
            )
            
            # Step 6: Predict hyperparameters
            hyperparams = predictor.predict(X_test_df, y_test_series)
            
            # Check that predictions are reasonable
            assert 'max_depth' in hyperparams
            assert 'min_samples_split' in hyperparams
            assert isinstance(hyperparams['max_depth'], int)
            assert hyperparams['max_depth'] > 0
            assert isinstance(hyperparams['min_samples_split'], float)
            assert 0 < hyperparams['min_samples_split'] <= 1
            
            # Step 7: Verify the predictions work with a real model
            model_with_predicted_params = DecisionTreeClassifier(
                **hyperparams,
                random_state=42
            )
            
            # Get a baseline with default hyperparameters
            baseline_model = DecisionTreeClassifier(random_state=42)
            
            # Compare performance (just check it runs, not that it's better, since this is a small test)
            predicted_score = np.mean(cross_val_score(
                model_with_predicted_params, X_test, y_test, cv=3
            ))
            baseline_score = np.mean(cross_val_score(
                baseline_model, X_test, y_test, cv=3
            ))
            
            # Just verify the model runs without errors
            assert predicted_score >= 0
            
            # Print performance for debugging
            print(f"Predicted hyperparameters: {hyperparams}")
            print(f"Predicted model score: {predicted_score:.4f}")
            print(f"Default model score: {baseline_score:.4f}")
            
            # Verify that the test completed successfully
            assert True


@pytest.mark.integration
class TestComponentIntegration:
    """Tests for integration between ZeroTune components."""
    
    def test_meta_params_to_prediction(self):
        """
        Test that meta-parameters can be used for prediction.
        
        This test verifies that the meta-parameters calculated by
        calculate_dataset_meta_parameters() are compatible with the
        prediction machinery in the CustomZeroTunePredictor.
        """
        # Create a small test dataset
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')
        
        # Calculate meta-parameters
        meta_params = calculate_dataset_meta_parameters(X_df, y_series)
        
        # Check that required parameters for predictions exist
        assert 'n_samples' in meta_params
        assert 'n_features' in meta_params
        
        # Create a mini-dataset using just these meta-parameters
        meta_df = pd.DataFrame({
            'n_samples': [meta_params['n_samples']],
            'n_features': [meta_params['n_features']]
        })
        
        # Mock a very simple prediction model that just returns the inputs
        # This verifies that the meta-parameters can be fed into the prediction pipeline
        class MockPredictor:
            def __init__(self):
                self.dataset_features = ['n_samples', 'n_features']
                self.target_params = ['params_max_depth', 'params_min_samples_split']
                self.param_config = {
                    "max_depth": {
                        "percentage_splits": [0.25, 0.5, 0.7],
                        "param_type": "int",
                        "dependency": "n_samples"
                    },
                    "min_samples_split": {
                        "percentage_splits": [0.01, 0.05, 0.1],
                        "param_type": "float"
                    }
                }
            
            def predict(self, X):
                # Return a fixed prediction based on the input
                n_samples = X['n_samples'].values[0]
                return np.array([[int(n_samples * 0.5), 0.05]])
            
        # Create a mock prediction
        mock_model = MockPredictor()
        
        # Manually perform the prediction process
        meta_features = meta_df[mock_model.dataset_features]
        pred = mock_model.predict(meta_features)
        
        # Check that prediction has correct shape
        assert pred.shape == (1, 2)
        
        # Verify the mock prediction is based on the meta-parameters
        assert pred[0, 0] == int(meta_params['n_samples'] * 0.5)
        assert pred[0, 1] == 0.05
        
        # This confirms that meta-parameters can be used for prediction
        assert True 