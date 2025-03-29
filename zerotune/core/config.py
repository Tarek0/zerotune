"""
Configuration settings for ZeroTune.

This module centralizes all configurable parameters used across the ZeroTune 
package, making it easier to manage and modify settings.
"""

from typing import Dict, Any, Optional, List
import os
from pathlib import Path

# Base paths
DEFAULT_BASE_DIR = os.path.join(os.path.expanduser("~"), ".zerotune")
DEFAULT_KB_DIR = os.path.join(DEFAULT_BASE_DIR, "knowledge_base")

# Knowledge base settings
DEFAULT_KB_FILENAME = "kb_{model_type}.json"

# Model configuration defaults
DEFAULT_MODEL_TYPE = "xgboost"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ITER = 10

# Dataset settings
DEFAULT_CORRELATION_CUTOFF = 0.1

# Optimization settings
DEFAULT_N_NEIGHBORS = 3

# Paths
def get_knowledge_base_path(model_type: str = "default") -> str:
    """
    Get the path to the knowledge base file for a given model type.
    
    Args:
        model_type: The type of model (e.g., "xgboost", "random_forest")
        
    Returns:
        Path to the knowledge base file
    """
    os.makedirs(DEFAULT_KB_DIR, exist_ok=True)
    return os.path.join(DEFAULT_KB_DIR, DEFAULT_KB_FILENAME.format(model_type=model_type))

# Supported models and metrics
SUPPORTED_MODELS = ["decision_tree", "random_forest", "xgboost"]
SUPPORTED_METRICS = ["accuracy", "f1", "mse", "rmse", "r2", "roc_auc"]

# Default hyperparameter optimization settings
DEFAULT_PARAM_GRID_SETTINGS = {
    "percentage_splits": {
        "small": [0.01, 0.05, 0.1, 0.2],
        "medium": [0.1, 0.3, 0.5, 0.7],
        "large": [0.5, 0.7, 0.8, 0.9, 0.99]
    },
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "regularization": [0.01, 0.1, 1.0, 10.0]
}

# Data preparation settings
DATA_PREPARATION = {
    "fill_missing": True,
    "categorical_strategy": "code",  # Options: "code", "onehot"
    "missing_numeric_fill": "mean",  # Options: "mean", "median", "zero"
    "missing_categorical_fill": -1   # Value to fill missing categorical values
}

# Recommended datasets by task
RECOMMENDED_DATASETS = {
    "classification": {
        "binary": [31, 1462, 1590],  # credit-g, banknote, adult
        "multi_class": [61, 40975, 40982]  # iris, connect-4, fashion-mnist
    },
    "regression": [42, 531, 560]  # housing, houses, bike-sharing
}

# Export configuration as a dictionary for easier access
CONFIG = {
    "paths": {
        "base_dir": DEFAULT_BASE_DIR,
        "kb_dir": DEFAULT_KB_DIR
    },
    "knowledge_base": {
        "filename_template": DEFAULT_KB_FILENAME
    },
    "defaults": {
        "model_type": DEFAULT_MODEL_TYPE,
        "test_size": DEFAULT_TEST_SIZE,
        "random_state": DEFAULT_RANDOM_STATE,
        "n_iter": DEFAULT_N_ITER,
        "n_neighbors": DEFAULT_N_NEIGHBORS,
        "correlation_cutoff": DEFAULT_CORRELATION_CUTOFF
    },
    "supported": {
        "models": SUPPORTED_MODELS,
        "metrics": SUPPORTED_METRICS
    },
    "hyperparameters": DEFAULT_PARAM_GRID_SETTINGS,
    "data_preparation": DATA_PREPARATION,
    "recommended_datasets": RECOMMENDED_DATASETS
} 