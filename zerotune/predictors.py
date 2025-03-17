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
        "description": "Decision Tree Classifier optimized for both binary and multi-class problems",
        "model_file_binary": "decision_tree_binary_classifier.joblib",
        "model_file_multiclass": "decision_tree_multiclass_classifier.joblib",
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
        "description": "Random Forest Classifier optimized for both binary and multi-class problems",
        "model_file_binary": "random_forest_binary_classifier.joblib",
        "model_file_multiclass": "random_forest_multiclass_classifier.joblib",
        "dataset_features": ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"],
        "param_config": {
            "n_estimators": {
                "percentage_splits": [0.1, 0.3, 0.5, 0.7, 0.9], 
                "param_type": "int", 
                "dependency": "n_samples",
                "multiplier": 2  # Adjusting based on dataset size
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
            },
            "max_features": {
                "percentage_splits": [0.5, 0.7, 0.8, 0.9, 0.99], 
                "param_type": "float"
            }
        }
    },
    "xgboost": {
        "description": "XGBoost Classifier optimized for both binary and multi-class problems",
        "model_file_binary": "xgboost_binary_classifier.joblib",
        "model_file_multiclass": "xgboost_multiclass_classifier.joblib",
        "dataset_features": ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"],
        "param_config": {
            "n_estimators": {
                "percentage_splits": [0.1, 0.3, 0.5, 0.7, 0.9], 
                "param_type": "int", 
                "dependency": "n_samples",
                "multiplier": 2  # Adjusting based on dataset size
            },
            "max_depth": {
                "percentage_splits": [0.2, 0.35, 0.5, 0.65, 0.8], 
                "param_type": "int", 
                "dependency": "n_samples"
            },
            "learning_rate": {
                "percentage_splits": [0.01, 0.05, 0.1, 0.2, 0.3],
                "param_type": "float",
                "is_direct": True  # Direct values, not percentages
            },
            "subsample": {
                "percentage_splits": [0.5, 0.7, 0.8, 0.9, 1.0],
                "param_type": "float",
                "is_direct": True
            },
            "colsample_bytree": {
                "percentage_splits": [0.5, 0.7, 0.8, 0.9, 1.0],
                "param_type": "float",
                "is_direct": True
            },
            "gamma": {
                "percentage_splits": [0, 0.1, 0.2, 0.5, 1.0],
                "param_type": "float",
                "is_direct": True
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

def _get_model_path(model_name: str, is_multiclass: bool = False) -> str:
    """Get the path to a pre-trained model file.
    
    Args:
        model_name: Name of the pre-trained model.
        is_multiclass: Whether to use the multi-class version of the model.
        
    Returns:
        Path to the model file.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
    
    # This will be used to locate files within the package
    if "model_file_binary" in AVAILABLE_MODELS[model_name]:
        # Model has separate binary and multi-class versions
        model_file = AVAILABLE_MODELS[model_name]["model_file_multiclass" if is_multiclass else "model_file_binary"]
    else:
        # Model has a single version for both binary and multi-class
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
        
        # Determine if the model has separate binary and multi-class versions
        self.has_separate_models = "model_file_binary" in self.model_info and "model_file_multiclass" in self.model_info
        
        # For models with separate versions, load both binary and multi-class models
        if self.has_separate_models:
            # Load binary model
            try:
                binary_model_path = _get_model_path(model_name, is_multiclass=False)
                binary_model_data = joblib.load(binary_model_path)
                self.binary_model = binary_model_data.get('model')
                self.binary_target_params = binary_model_data.get('target_params', [])
                
                # Get normalization parameters for binary model
                if 'normalization_params' in binary_model_data and binary_model_data['normalization_params']:
                    self.binary_normalization_params = binary_model_data['normalization_params']
                else:
                    self.binary_normalization_params = self._create_identity_normalization(self.binary_target_params)
                
                # Load multi-class model
                multiclass_model_path = _get_model_path(model_name, is_multiclass=True)
                multiclass_model_data = joblib.load(multiclass_model_path)
                self.multiclass_model = multiclass_model_data.get('model')
                self.multiclass_target_params = multiclass_model_data.get('target_params', [])
                
                # Get normalization parameters for multi-class model
                if 'normalization_params' in multiclass_model_data and multiclass_model_data['normalization_params']:
                    self.multiclass_normalization_params = multiclass_model_data['normalization_params']
                else:
                    self.multiclass_normalization_params = self._create_identity_normalization(self.multiclass_target_params)
                
                # Default model, target params, and normalization params (for compatibility)
                self.model = self.binary_model
                self.target_params = self.binary_target_params
                self.normalization_params = self.binary_normalization_params
                
            except (FileNotFoundError, KeyError, ValueError) as e:
                # If loading separate models fails, fall back to a single model
                self.has_separate_models = False
                print(f"Warning: Failed to load separate models for {model_name}: {str(e)}")
                print("Falling back to single model mode.")
        
        # For models with a single version (or fallback), load the model
        if not self.has_separate_models:
            try:
                model_path = _get_model_path(model_name)
                model_data = joblib.load(model_path)
                
                # Check if model_data is already a sklearn model (for backward compatibility)
                if hasattr(model_data, 'predict'):
                    self.model = model_data
                    self.target_params = self.dataset_features  # Fallback
                    # Create identity normalization for backward compatibility with old models
                    self.normalization_params = self._create_identity_normalization(self.target_params)
                else:
                    # Expect a dictionary with 'model', 'target_params', and 'normalization_params'
                    self.model = model_data.get('model')
                    self.target_params = model_data.get('target_params', [])
                    
                    # Get normalization parameters or create identity normalization if not present
                    if 'normalization_params' in model_data and model_data['normalization_params']:
                        self.normalization_params = model_data['normalization_params']
                    else:
                        self.normalization_params = self._create_identity_normalization(self.target_params)
                    
                if self.model is None:
                    raise KeyError("Model not found in loaded data")
                    
            except (FileNotFoundError, KeyError) as e:
                # If the pre-trained model isn't available, provide instructions
                raise RuntimeError(
                    f"Pre-trained model '{model_name}' could not be loaded: {str(e)}\n"
                    "You can train your own model using the KnowledgeBase class."
                )
    
    def _create_identity_normalization(self, target_params):
        """Create identity normalization parameters for backward compatibility.
        
        Args:
            target_params: List of target parameter names.
            
        Returns:
            Dictionary with identity normalization parameters.
        """
        normalization_params = {}
        for param in target_params:
            normalization_params[param] = {
                'min': 0,
                'max': 1,
                'range': 1
            }
        return normalization_params
    
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
        
        # Determine if this is a multi-class problem
        is_multiclass = len(np.unique(y)) > 2
        meta_params['n_classes'] = len(np.unique(y))
        
        # For models with separate binary and multi-class versions, select the appropriate one
        if self.has_separate_models:
            if is_multiclass:
                model = self.multiclass_model
                target_params = self.multiclass_target_params
                normalization_params = self.multiclass_normalization_params
                print(f"Using multi-class model for dataset with {meta_params.get('n_classes', '?')} classes")
            else:
                model = self.binary_model
                target_params = self.binary_target_params
                normalization_params = self.binary_normalization_params
                print(f"Using binary model for dataset with {meta_params.get('n_classes', '?')} classes")
        else:
            # Use the single model
            model = self.model
            target_params = self.target_params
            normalization_params = self.normalization_params
        
        # Extract only the needed features
        meta_features = pd.DataFrame([{key: meta_params.get(key, 0) for key in self.dataset_features}])
        
        # Make prediction
        # Handle potential feature name mismatch by using the model directly without feature names check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(model, 'feature_names_in_'):
                # For newer sklearn versions that validate feature names, use a numpy array instead
                predictions = model.predict(meta_features.values)
            else:
                predictions = model.predict(meta_features)
        
        # Make sure we have a 2D array of predictions
        if len(predictions.shape) == 1:
            predictions = [predictions]
            
        # Create predictions dictionary with denormalization
        predictions_dict = {}
        for i, target_param in enumerate(target_params):
            if i < len(predictions[0]):
                # Apply denormalization using the normalization parameters
                params = normalization_params.get(target_param, {'min': 0, 'max': 1})
                if 'range' in params and params['range'] > 0:
                    pred_value = predictions[0][i] * params['range'] + params['min']
                else:
                    # Calculate range if not present
                    range_val = params.get('max', 1) - params.get('min', 0)
                    if range_val > 0:
                        pred_value = predictions[0][i] * range_val + params.get('min', 0)
                    else:
                        pred_value = params.get('min', 0)
                predictions_dict[target_param] = pred_value
        
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
        
        # Cap extreme hyperparameter values
        absolute_params = self._cap_extreme_values(absolute_params)
        
        # Define valid parameters for each model type
        valid_params_by_model = {
            'decision_tree': ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 
                              'criterion', 'splitter', 'max_leaf_nodes', 'random_state', 'ccp_alpha'],
            'random_forest': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features',
                              'criterion', 'max_leaf_nodes', 'bootstrap', 'random_state', 'n_jobs', 'oob_score'],
            'xgboost': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'gamma',
                        'min_child_weight', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs', 'tree_method']
        }
        
        # Get the appropriate valid parameters based on model name
        if self.model_name in valid_params_by_model:
            valid_params = valid_params_by_model[self.model_name]
        else:
            # Fallback to only the parameters in the param_config
            valid_params = list(self.param_config.keys())
        
        return {k: v for k, v in absolute_params.items() if k in valid_params}
    
    def _cap_extreme_values(self, params: Dict[str, float]) -> Dict[str, float]:
        """Cap extreme hyperparameter values to reasonable ranges.
        
        Args:
            params: Dictionary of hyperparameter values.
            
        Returns:
            Dictionary of capped hyperparameter values.
        """
        # Create a copy to avoid modifying the original
        capped_params = params.copy()
        
        # Define reasonable caps for each hyperparameter type
        caps = {
            # Tree-based models
            "n_estimators": 1000,        # Max 1000 trees for random forest/boosting
            "max_depth": 30,             # Max tree depth of 30
            "min_samples_split": 0.5,    # Max 50% of samples for split
            "min_samples_leaf": 0.5,     # Max 50% of samples per leaf
            "max_features": 1.0,         # Max 100% of features
            
            # XGBoost specific
            "learning_rate": 1.0,        # Max learning rate of 1.0
            "subsample": 1.0,            # Max subsample ratio of 1.0
            "colsample_bytree": 1.0,     # Max column subsample ratio of 1.0
            "gamma": 10.0,               # Max gamma of 10.0
            "reg_alpha": 10.0,           # Max L1 regularization of 10.0
            "reg_lambda": 10.0,          # Max L2 regularization of 10.0
            "min_child_weight": 10.0     # Max min_child_weight of 10.0
        }
        
        # Apply caps to each parameter
        for param, value in capped_params.items():
            if param in caps:
                capped_params[param] = min(value, caps[param])
                
                # Apply specific minimum values for certain parameters
                if param == "n_estimators":
                    capped_params[param] = max(10, capped_params[param])  # At least 10 estimators
                elif param == "max_depth":
                    capped_params[param] = max(1, capped_params[param])   # At least depth 1
        
        return capped_params
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        info = {
            "name": self.model_name,
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "description": self.model_info["description"],
            "dataset_features": self.dataset_features,
            "target_params": self.target_params
        }
        
        if self.has_separate_models:
            info["has_separate_models"] = True
            info["binary_model_type"] = type(self.binary_model).__name__
            info["multiclass_model_type"] = type(self.multiclass_model).__name__
            info["binary_target_params"] = self.binary_target_params
            info["multiclass_target_params"] = self.multiclass_target_params
        
        return info


class CustomZeroTunePredictor:
    """Class for making hyperparameter predictions using custom ZeroTune models."""
    
    def __init__(self, model_path: str, param_config: Dict):
        """Initialize a predictor with a custom model.
        
        Args:
            model_path: Path to the saved model file.
            param_config: Parameter configuration dictionary.
        """
        self.model_path = model_path
        self.param_config = param_config
        
        # Load the model
        try:
            model_data = joblib.load(model_path)
            
            # Check if model_data is already a sklearn model (for backward compatibility)
            if hasattr(model_data, 'predict'):
                self.model = model_data
                self.dataset_features = []  # Fallback to be defined later
                self.target_params = []  # Fallback to be defined later
                # Create identity normalization for old models
                self.normalization_params = {}  # Will be populated when target_params are known
            else:
                self.model = model_data.get('model')
                self.dataset_features = model_data.get('dataset_features', [])
                self.target_params = model_data.get('target_params', [])
                
                # Get normalization parameters or create identity normalization if not present
                if 'normalization_params' in model_data and model_data['normalization_params']:
                    self.normalization_params = model_data['normalization_params']
                else:
                    self.normalization_params = self._create_identity_normalization(self.target_params)
            
            if self.model is None:
                raise ValueError("Model not found in loaded data")
            
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def _create_identity_normalization(self, target_params):
        """Create identity normalization parameters for backward compatibility.
        
        Args:
            target_params: List of target parameter names.
            
        Returns:
            Dictionary with identity normalization parameters.
        """
        normalization_params = {}
        for param in target_params:
            normalization_params[param] = {
                'min': 0,
                'max': 1,
                'range': 1
            }
        return normalization_params
    
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
        
        # For old models that didn't provide target_params, create them now
        if not self.target_params and predictions.shape[1] > 0:
            self.target_params = [f"params_{param}" for param in self.param_config.keys()]
            # Now that we have target_params, create identity normalization
            if not self.normalization_params:
                self.normalization_params = self._create_identity_normalization(self.target_params)
            
        # Create predictions dictionary with denormalization
        predictions_dict = {}
        for i, target_param in enumerate(self.target_params):
            if i < len(predictions[0]):
                # Apply denormalization using the normalization parameters
                params = self.normalization_params.get(target_param, {'min': 0, 'max': 1, 'range': 1})
                if params['range'] > 0:
                    pred_value = predictions[0][i] * params['range'] + params['min']
                else:
                    pred_value = params['min']
                predictions_dict[target_param] = pred_value
        
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
        
        # Detect model type based on target parameters
        model_type = "unknown"
        # Check for XGBoost-specific parameters
        if any(param in ['learning_rate', 'subsample', 'colsample_bytree', 'gamma'] for param in self.param_config):
            model_type = "xgboost"
        # Check for Random Forest specific parameters (when we have n_estimators but not XGBoost params)
        elif 'n_estimators' in self.param_config:
            model_type = "random_forest"
        # Fallback to decision tree
        else:
            model_type = "decision_tree"
            
        # Define valid parameters for each model type
        valid_params_by_model = {
            'decision_tree': ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 
                              'criterion', 'splitter', 'max_leaf_nodes', 'random_state', 'ccp_alpha'],
            'random_forest': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features',
                              'criterion', 'max_leaf_nodes', 'bootstrap', 'random_state', 'n_jobs', 'oob_score'],
            'xgboost': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'gamma',
                        'min_child_weight', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs', 'tree_method']
        }
        
        # Get the appropriate valid parameters
        if model_type in valid_params_by_model:
            valid_params = valid_params_by_model[model_type]
        else:
            # Fallback to all parameters
            valid_params = list(self.param_config.keys())
            
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