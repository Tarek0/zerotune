"""
Model configuration classes for ZeroTune.

This module contains the configurations for different machine learning models
that can be optimized using ZeroTune's meta-learning approach.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelConfigs:
    """
    Configuration class for model hyperparameters.
    
    This class provides static methods to retrieve standardized configurations
    for different machine learning models, including default instances and
    hyperparameter search spaces.
    """
    
    @staticmethod
    def get_decision_tree_config():
        """
        Returns configuration for DecisionTreeClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            dict: Configuration dictionary with keys:
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
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "int", 'dependency': 'n_samples'},
                'min_samples_split': {'percentage_splits': [0.005, 0.01, 0.02, 0.05, 0.10], 'param_type': "float"},
                'min_samples_leaf': {'percentage_splits': [0.005, 0.01, 0.02, 0.05, 0.10], 'param_type': "float"},
                'max_features': {'percentage_splits': [0.50, 0.70, 0.8, 0.9, 0.99], 'param_type': "float"}
            }
        }
    
    @staticmethod
    def get_random_forest_config():
        """
        Returns configuration for RandomForestClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            dict: Configuration dictionary with keys:
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
                'n_estimators': {'percentage_splits': [0.1, 0.2, 0.3, 0.4, 0.5], 'param_type': "int", 'dependency': 'n_features', 'multiplier': 100},
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "int", 'dependency': 'n_samples'},
                'min_samples_split': {'percentage_splits': [0.005, 0.01, 0.02, 0.05, 0.10], 'param_type': "float"},
                'min_samples_leaf': {'percentage_splits': [0.005, 0.01, 0.02, 0.05, 0.10], 'param_type': "float"},
                'max_features': {'percentage_splits': [0.50, 0.70, 0.8, 0.9, 0.99], 'param_type': "float"}
            }
        }
    
    @staticmethod
    def get_xgboost_config():
        """
        Returns configuration for XGBClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            dict: Configuration dictionary with keys:
                - name: Name of the model
                - model: Instance of XGBClassifier
                - metric: Evaluation metric to use
                - param_config: Dictionary of hyperparameters and their search ranges
        """
        return {
            "name": "XGBClassifier",
            "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'metric': 'roc_auc',
            'param_config': {
                'n_estimators': {'percentage_splits': [0.1, 0.2, 0.3, 0.4, 0.5], 'param_type': "int", 'dependency': 'n_features', 'multiplier': 100},
                'max_depth': {'percentage_splits': [0.1, 0.2, 0.3, 0.4, 0.5], 'param_type': "int", 'dependency': 'n_samples'},
                'learning_rate': {'percentage_splits': [0.01, 0.05, 0.1, 0.2, 0.3], 'param_type': "float"},
                'subsample': {'percentage_splits': [0.5, 0.6, 0.7, 0.8, 0.9], 'param_type': "float"},
                'colsample_bytree': {'percentage_splits': [0.5, 0.6, 0.7, 0.8, 0.9], 'param_type': "float"}
            }
        } 