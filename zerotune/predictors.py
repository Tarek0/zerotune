"""
ZeroTune Predictor for using pre-trained models.

This module provides the ZeroTunePredictor class that uses pre-trained models
to predict optimal hyperparameters for new datasets.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union

from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
from zerotune.core.utils import convert_to_dataframe
from zerotune.core.predictor_training import train_predictor_from_knowledge_base


class ZeroTunePredictor:
    """
    Predictor class that uses pre-trained models for zero-shot hyperparameter prediction.
    """
    
    def __init__(self, model_name: str, task_type: str = "binary"):
        """
        Initialize the predictor with a pre-trained model.
        
        Args:
            model_name: Name of the model ("decision_tree", "random_forest", "xgboost")
            task_type: Type of task ("binary", "multiclass", "regression")
        """
        self.model_name = model_name
        self.task_type = task_type
        self.model_data = None
        self.norm_params = None
        
        # Load the pre-trained model
        self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load the pre-trained model from the models directory."""
        # Construct the model filename
        model_filename = f"{self.model_name}_{self.task_type}_classifier.joblib"
        if self.task_type == "regression":
            model_filename = f"{self.model_name}_regressor.joblib"
        
        # Get the path to the models directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(models_dir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pre-trained model not found: {model_path}")
        
        # Load the model data
        self.model_data = joblib.load(model_path)
        
        # Store normalization parameters for manual scaling
        if 'normalization_params' in self.model_data:
            self.norm_params = self.model_data['normalization_params']
        else:
            self.norm_params = None
    
    @classmethod
    def train_from_knowledge_base(
        cls,
        kb_path: str,
        model_name: str,
        task_type: str = "binary",
        output_dir: str = "models",
        exp_id: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> 'ZeroTunePredictor':
        """
        Train a new zero-shot predictor from a knowledge base.
        
        Args:
            kb_path: Path to the knowledge base JSON file
            model_name: Name of the target model ("xgboost", "random_forest", etc.)
            task_type: Type of task ("binary", "multiclass", "regression")
            output_dir: Directory to save the trained predictor
            exp_id: Experiment ID for naming (extracted from kb_path if None)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Trained ZeroTunePredictor instance
        """
        # Use the core training function
        model_path = train_predictor_from_knowledge_base(
            kb_path=kb_path,
            model_name=model_name,
            task_type=task_type,
            output_dir=output_dir,
            exp_id=exp_id,
            test_size=test_size,
            random_state=random_state,
            verbose=True
        )
        
        # Load the trained model and create a ZeroTunePredictor instance
        model_data = joblib.load(model_path)
        
        # Create and return a ZeroTunePredictor instance with the new model
        predictor = cls.__new__(cls)
        predictor.model_name = model_name
        predictor.task_type = task_type
        predictor.model_data = model_data
        predictor.norm_params = model_data['normalization_params']
        
        return predictor

    def predict(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Predict optimal hyperparameters for the given dataset.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of optimal hyperparameters
        """
        if self.model_data is None:
            raise RuntimeError("No pre-trained model loaded")
        
        # Convert to DataFrame if needed
        X_df = convert_to_dataframe(X)
        
        # Calculate meta-features for the dataset
        meta_features = calculate_dataset_meta_parameters(X_df, y)
        
        # Extract only the features that the pre-trained model expects
        feature_names = self.model_data['dataset_features']
        feature_vector = []
        
        for feature_name in feature_names:
            if feature_name in meta_features:
                feature_vector.append(float(meta_features[feature_name]))
            else:
                print(f"Warning: Feature '{feature_name}' not found in meta-features, using 0.0")
                feature_vector.append(0.0)
        
        # Convert to numpy array and reshape for prediction
        feature_vector = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
        
        # Normalize features if normalization parameters are available
        if self.norm_params is not None:
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.norm_params:
                    norm_info = self.norm_params[feature_name]
                    min_val = norm_info.get('min', 0.0)
                    max_val = norm_info.get('max', 1.0)
                    range_val = norm_info.get('range', 1.0)
                    
                    # Normalize to [0, 1] range using min-max scaling
                    if range_val > 0:
                        feature_vector[0, i] = (feature_vector[0, i] - min_val) / range_val
                    else:
                        feature_vector[0, i] = 0.0
        
        # Make prediction using the pre-trained model
        prediction = self.model_data['model'].predict(feature_vector)[0]
        
        # Convert prediction to hyperparameter dictionary
        target_params = self.model_data['target_params']
        hyperparams = {}
        
        for i, param_name in enumerate(target_params):
            # Remove 'params_' prefix if present
            clean_param_name = param_name.replace('params_', '')
            
            if i < len(prediction):
                value = prediction[i]
                
                # Convert to appropriate type based on parameter
                if clean_param_name in ['n_estimators', 'max_depth']:
                    value = max(1, int(round(value)))
                elif clean_param_name in ['learning_rate', 'subsample', 'colsample_bytree', 'gamma']:
                    value = max(0.01, min(1.0, float(value)))
                
                hyperparams[clean_param_name] = value
        
        return hyperparams


# Available pre-trained models
AVAILABLE_MODELS = {
    'decision_tree': {
        'binary': 'decision_tree_binary_classifier.joblib',
        'multiclass': 'decision_tree_multiclass_classifier.joblib',
        'regression': 'decision_tree_regressor.joblib'
    },
    'random_forest': {
        'binary': 'random_forest_binary_classifier.joblib',
        'multiclass': 'random_forest_multiclass_classifier.joblib',
        'regression': 'random_forest_regressor.joblib'
    },
    'xgboost': {
        'binary': 'xgboost_binary_classifier.joblib',
        'multiclass': 'xgboost_multiclass_classifier.joblib',
        'regression': 'xgboost_regressor.joblib'
    }
} 