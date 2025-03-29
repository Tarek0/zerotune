"""
Utility functions for ZeroTune.

This module contains various utility functions used across the ZeroTune package.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Type
import numpy as np
import pandas as pd
import json
import os
from sklearn.base import BaseEstimator

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
HyperParams = Dict[str, Any]
MetaFeatures = Dict[str, float]


def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize objects to JSON, handling NumPy types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj


def load_json(file_path: str, default: Optional[Any] = None) -> Any:
    """
    Load a JSON file safely with error handling.
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if loading fails
        
    Returns:
        Loaded JSON data or default value if loading failed
    """
    if not os.path.exists(file_path):
        return default
        
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        return default


def save_json(data: Any, file_path: str, ensure_dir: bool = True) -> bool:
    """
    Save data to a JSON file with proper serialization.
    
    Args:
        data: Data to save (will be serialized using safe_json_serialize)
        file_path: Path to save the JSON file
        ensure_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        True if saving was successful, False otherwise
    """
    if ensure_dir and os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, default=safe_json_serialize, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {str(e)}")
        return False


def is_numeric_dtype(dtype) -> bool:
    """
    Check if a pandas dtype is numeric.
    
    Args:
        dtype: pandas or numpy dtype to check
        
    Returns:
        True if the dtype is numeric (int or float), False otherwise
    """
    return pd.api.types.is_numeric_dtype(dtype)


def select_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only numeric columns from a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with only numeric columns
    """
    return df.select_dtypes(include=['int64', 'float64'])


def is_classification_task(y: ArrayLike) -> bool:
    """
    Determine if a target variable represents a classification task.
    
    Args:
        y: Target variable to check
        
    Returns:
        True if the task appears to be classification, False otherwise
    """
    if hasattr(y, 'dtype'):
        # If categorical dtype, it's classification
        if pd.api.types.is_categorical_dtype(y.dtype):
            return True
        
        # If object dtype, check if it contains strings
        if y.dtype == 'object':
            return True
            
        # Check number of unique values
        try:
            n_unique = len(np.unique(y))
            return n_unique < 0.2 * len(y)  # Heuristic for classification
        except:
            pass
    
    return False  # Default to regression
        

def make_base_params_dict(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Create a basic parameters dictionary for models.
    
    Args:
        random_state: Random state to use for all models
        
    Returns:
        Dictionary of base parameters for common model types
    """
    return {
        "decision_tree": {"random_state": random_state},
        "random_forest": {"random_state": random_state},
        "xgboost": {
            "random_state": random_state,
            "eval_metric": "logloss",
            "enable_categorical": True
        }
    }


def convert_to_dataframe(X: ArrayLike) -> pd.DataFrame:
    """
    Convert array-like to DataFrame if it's not already one.
    
    Args:
        X: Array-like object (numpy array, DataFrame, etc.)
        
    Returns:
        Pandas DataFrame version of the input
    """
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    else:
        return pd.DataFrame(np.array(X)) 