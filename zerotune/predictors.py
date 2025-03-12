"""Predictors module for ZeroTune.

This module provides pre-trained models for zero-shot hyperparameter optimization.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import pkg_resources
import warnings

from .zerotune import (
    calculate_dataset_meta_parameters,
    relative2absolute_dict,
    remove_param_prefix
)

# Dictionary of available pre-trained models with their configurations
AVAILABLE_MODELS = {
    "decision_tree": {
        "description": "Decision Tree Classifier with 4 hyperparameters",
        "model_file": "decision_tree_classifier.joblib",
        "dataset_features": ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"],
        "param_config": {
            "max_depth": {
                "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999], 
                "param_type": "int", 
                "dependency": "n_samples"
            },
            "min_samples_split": {
                "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
                "param_type": "float"
            },
            "min_samples_leaf": {
                "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
                "param_type": "float"
            },
            "max_features": {
                "percentage_splits": [0.5, 0.7, 0.8, 0.9, 0.99], 
                "param_type": "float"
            }
        }
    },
    "random_forest": {
        "description": "Random Forest Classifier with key hyperparameters",
        "model_file": "random_forest_classifier.joblib",
        "dataset_features": ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"],
        "param_config": {
            "n_estimators": {
                "percentage_splits": [0.1, 0.3, 0.5, 0.7, 0.9], 
                "param_type": "int", 
                "dependency": "n_samples"
            },
            "max_depth": {
                "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999], 
                "param_type": "int", 
                "dependency": "n_samples"
            },
            "min_samples_split": {
                "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
                "param_type": "float"
            },
            "min_samples_leaf": {
                "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
                "param_type": "float"
            }
        }
    }
}

def get_available_models() -> List[str]:
    """Get information about available pre-trained models.
    
    Returns:
        List of available model names.
    """
    return list(AVAILABLE_MODELS.keys())

def _get_model_path(model_name: str) -> str:
    """Get the path to a pre-trained model file.
    
    Args:
        model_name: Name of the pre-trained model.
        
    Returns:
        Path to the model file.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
    
    # This will be used to locate files within the package
    model_file = AVAILABLE_MODELS[model_name]["model_file"]
    model_path = pkg_resources.resource_filename("zerotune", f"models/{model_file}")
    
    # For development, if the model doesn't exist in the package resources,
    # look for it in the current directory
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "models", model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return model_path

class ZeroTunePredictor:
    """Class for making hyperparameter predictions using pre-trained ZeroTune models."""
    
    def __init__(self, model_name: str = "decision_tree"):
        """Initialize a predictor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use.
        """
        self.model_name = model_name
        
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not found. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        
        self.model_info = AVAILABLE_MODELS[model_name]
        self.param_config = self.model_info["param_config"]
        self.dataset_features = self.model_info["dataset_features"]
        
        # Load the model
        try:
            model_path = _get_model_path(model_name)
            model_data = joblib.load(model_path)
            
            # Check if model_data is already a sklearn model (for backward compatibility)
            if hasattr(model_data, 'predict'):
                self.model = model_data
                self.target_params = self.dataset_features  # Fallback
            else:
                # Expect a dictionary with 'model' and 'target_params'
                self.model = model_data.get('model')
                self.target_params = model_data.get('target_params', [])
                
            if self.model is None:
                raise KeyError("Model not found in loaded data")
                
        except (FileNotFoundError, KeyError) as e:
            # If the pre-trained model isn't available, provide instructions
            raise RuntimeError(
                f"Pre-trained model '{model_name}' could not be loaded: {str(e)}\n"
                "You can train your own model using the KnowledgeBase class."
            )
    
    def predict(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Predict hyperparameters for a dataset.
        
        Args:
            X: Feature data.
            y: Target data.
            
        Returns:
            Dictionary of predicted hyperparameters.
        """
        # Calculate dataset meta-parameters
        meta_params = calculate_dataset_meta_parameters(X, y)
        
        # Extract only the needed features
        meta_features = pd.DataFrame([{key: meta_params.get(key, 0) for key in self.dataset_features}])
        
        # Make prediction
        # Handle potential feature name mismatch by using the model directly without feature names check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(self.model, 'feature_names_in_'):
                # For newer sklearn versions that validate feature names, use a numpy array instead
                predictions = self.model.predict(meta_features.values)
            else:
                predictions = self.model.predict(meta_features)
        
        # Make sure we have a 2D array of predictions
        if len(predictions.shape) == 1:
            predictions = [predictions]
            
        # Create predictions dictionary
        predictions_dict = {}
        for i, target_param in enumerate(self.target_params):
            if i < len(predictions[0]):
                predictions_dict[target_param] = predictions[0][i]
        
        # Remove 'params_' prefix if present
        predictions_dict = remove_param_prefix(predictions_dict)
        
        # Make sure all parameters in param_config are present in predictions_dict
        for param in self.param_config:
            if param not in predictions_dict:
                # Use a default value based on parameter config
                if 'default' in self.param_config[param]:
                    predictions_dict[param] = self.param_config[param]['default']
                elif 'percentage_splits' in self.param_config[param]:
                    # Use middle value from percentage splits
                    splits = self.param_config[param]['percentage_splits']
                    predictions_dict[param] = splits[len(splits) // 2]
        
        # Convert relative parameters to absolute where needed
        absolute_params = relative2absolute_dict(self.param_config, meta_params, predictions_dict)
        
        # Filter out any parameters that aren't valid for DecisionTreeClassifier
        valid_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 
                        'criterion', 'splitter', 'max_leaf_nodes', 'random_state', 'ccp_alpha']
        return {k: v for k, v in absolute_params.items() if k in valid_params}
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "name": self.model_name,
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "description": self.model_info["description"],
            "dataset_features": self.dataset_features,
            "target_params": self.target_params
        }


class CustomZeroTunePredictor:
    """Class for making hyperparameter predictions using custom ZeroTune models."""
    
    def __init__(self, model_path: str, param_config: Dict):
        """Initialize a predictor with a custom trained model.
        
        Args:
            model_path: Path to the saved model file.
            param_config: Parameter configuration for conversion to absolute values.
        """
        self.model_path = model_path
        self.param_config = param_config
        
        # Load the model
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.dataset_features = model_data['dataset_features']
            self.target_params = model_data['target_params']
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"Model could not be loaded from {model_path}: {str(e)}")
    
    def predict(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Predict hyperparameters for a dataset.
        
        Args:
            X: Feature data.
            y: Target data.
            
        Returns:
            Dictionary of predicted hyperparameters.
        """
        # Calculate dataset meta-parameters
        meta_params = calculate_dataset_meta_parameters(X, y)
        
        # Extract only the needed features
        meta_features = pd.DataFrame([{key: meta_params.get(key, 0) for key in self.dataset_features}])
        
        # Make prediction
        # Handle potential feature name mismatch by using the model directly without feature names check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(self.model, 'feature_names_in_'):
                # For newer sklearn versions that validate feature names, use a numpy array instead
                predictions = self.model.predict(meta_features.values)
            else:
                predictions = self.model.predict(meta_features)
        
        # Make sure we have a 2D array of predictions
        if len(predictions.shape) == 1:
            predictions = [predictions]
            
        # Create predictions dictionary
        predictions_dict = {}
        for i, target_param in enumerate(self.target_params):
            if i < len(predictions[0]):
                predictions_dict[target_param] = predictions[0][i]
        
        # Remove 'params_' prefix if present
        predictions_dict = remove_param_prefix(predictions_dict)
        
        # Make sure all parameters in param_config are present in predictions_dict
        for param in self.param_config:
            if param not in predictions_dict:
                # Use a default value based on parameter config
                if 'default' in self.param_config[param]:
                    predictions_dict[param] = self.param_config[param]['default']
                elif 'percentage_splits' in self.param_config[param]:
                    # Use middle value from percentage splits
                    splits = self.param_config[param]['percentage_splits']
                    predictions_dict[param] = splits[len(splits) // 2]
        
        # Convert relative parameters to absolute where needed
        absolute_params = relative2absolute_dict(self.param_config, meta_params, predictions_dict)
        
        # Filter out any parameters that aren't valid for DecisionTreeClassifier
        valid_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 
                        'criterion', 'splitter', 'max_leaf_nodes', 'random_state', 'ccp_alpha']
        return {k: v for k, v in absolute_params.items() if k in valid_params}
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_path": self.model_path,
            "dataset_features": self.dataset_features,
            "target_params": self.target_params
        } 