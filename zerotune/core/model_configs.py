"""
Model configuration classes for ZeroTune.

This module contains the configurations for different machine learning models
that can be optimized using ZeroTune's meta-learning approach.
"""

from typing import Dict, List, Union, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Type aliases
ModelConfig = Dict[str, Any]
ParamConfig = Dict[str, Dict[str, Union[List[float], str, float]]]


class ModelConfigs:
    """
    Configuration class for model hyperparameters.
    
    This class provides static methods to retrieve standardized configurations
    for different machine learning models, including default instances and
    hyperparameter search spaces.
    """
    
    @staticmethod
    def get_decision_tree_config() -> ModelConfig:
        """
        Returns configuration for DecisionTreeClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            Configuration dictionary with keys:
                - name: Name of the model
                - model: Instance of DecisionTreeClassifier
                - metric: Evaluation metric to use
                - param_config: Dictionary of hyperparameters and their search ranges
        """
        return {
            "name": "DecisionTreeClassifier",
            "model": DecisionTreeClassifier(random_state=42),
            'metric': 'roc_auc',
            'param_config': {
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "float", 'dependency': 'n_samples'},
                'min_samples_split': {'min_value': 0.01, 'max_value': 0.5, 'param_type': "float"},  # 1% to 50% of samples
                'min_samples_leaf': {'min_value': 0.005, 'max_value': 0.2, 'param_type': "float"},  # 0.5% to 20% of samples
                'max_features': {'min_value': 0.1, 'max_value': 1.0, 'param_type': "float"}  # 10% to 100% of features
            }
        }
    
    @staticmethod
    def get_random_forest_config() -> ModelConfig:
        """
        Returns configuration for RandomForestClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            Configuration dictionary with keys:
                - name: Name of the model
                - model: Instance of RandomForestClassifier
                - metric: Evaluation metric to use
                - param_config: Dictionary of hyperparameters and their search ranges
        """
        return {
            "name": "RandomForestClassifier",
            "model": RandomForestClassifier(random_state=42),
            'metric': 'roc_auc',
            'param_config': {
                'n_estimators': {'min_value': 10, 'max_value': 250, 'param_type': "int"},
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "float", 'dependency': 'n_samples'},
                'min_samples_split': {'percentage_splits': [0.01, 0.02, 0.05, 0.10, 0.20], 'param_type': "float"},
                'min_samples_leaf': {'percentage_splits': [0.005, 0.01, 0.02, 0.05, 0.10], 'param_type': "float"},
                'max_features': {'percentage_splits': [0.50, 0.70, 0.8, 0.9, 0.99], 'param_type': "float"}
            }
        }
    
    @staticmethod
    def get_xgboost_config() -> ModelConfig:
        """
        Returns configuration for XGBClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            Configuration dictionary with keys:
                - name: Name of the model
                - model: Instance of XGBClassifier
                - metric: Evaluation metric to use
                - param_config: Dictionary of hyperparameters and their search ranges
        """
        return {
            "name": "XGBClassifier",
            "model": XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                enable_categorical=True
            ),
            'metric': 'roc_auc',
            'param_config': {
                'n_estimators': {'min_value': 10, 'max_value': 250, 'param_type': "int"},
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "float", 'dependency': 'n_samples'},
                'learning_rate': {'min_value': 0.001, 'max_value': 0.5, 'param_type': "float"},
                'subsample': {'min_value': 0.5, 'max_value': 1.0, 'param_type': "float"},
                'colsample_bytree': {'min_value': 0.5, 'max_value': 1.0, 'param_type': "float"}
            }
        } 