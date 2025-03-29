"""
Data loading functions for ZeroTune.

This module handles loading datasets from various sources including OpenML
and preparing them for use with ZeroTune.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import openml
import os
import json

# Type aliases
DatasetDict = Dict[str, Dict[str, Union[int, Dict[str, Any]]]]
DatasetTuple = Tuple[pd.DataFrame, str, str]
ProcessedData = Tuple[pd.DataFrame, pd.Series]


def fetch_open_ml_data(dataset_id: int) -> DatasetTuple:
    """
    Fetches a dataset from OpenML by ID.
    
    Args:
        dataset_id: The OpenML dataset ID
        
    Returns:
        A tuple containing:
            - dataframe: The loaded dataset as a pandas DataFrame
            - target_name: Name of the target column
            - dataset_name: Name of the dataset
    """
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True
    )
    print(f'Dataset name: {dataset.name}')
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )
    X["target"] = y
    
    return X, 'target', dataset.name


def prepare_data(df: pd.DataFrame, target_name: str) -> ProcessedData:
    """
    Simple preprocessing wrapper function that prepares data for machine learning.
    
    Args:
        df: Pandas dataframe containing dataset
        target_name: The name of the target variable column
        
    Returns:
        A tuple containing:
            - X: Feature matrix as DataFrame with:
                - Categorical/string columns converted to numeric codes
                - Missing numeric values filled with column means
                - Remaining missing values filled with -1
            - y: Target variable as Series (converted to int if categorical)
    """
    # Extract target and convert to numeric if needed
    y = df[target_name]
    if y.dtype == 'object' or y.dtype == 'category':
        # Convert categorical target to numeric
        y = pd.factorize(y)[0]  # Returns (codes, unique_values)
    
    # Extract features
    X = df.drop(target_name, axis=1)
    
    # Handle categorical and string columns
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            # Convert to category type for XGBoost
            X[col] = pd.Categorical(X[col]).codes
    
    # Fill missing values
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(-1)  # Fill remaining missing values with -1
    
    return X, y


def load_dataset_catalog(json_path: Optional[str] = None) -> DatasetDict:
    """
    Load the dataset catalog from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing dataset information.
            If None, the default openml_datasets.json in the package root is used.
            
    Returns:
        A dictionary with dataset categories and their OpenML dataset IDs
    """
    if json_path is None:
        # Try to locate the default JSON file in the package root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(root_dir, 'openml_datasets.json')
    
    if not os.path.exists(json_path):
        print(f"Warning: Dataset catalog file not found at {json_path}")
        return {
            "binary": {},
            "multi-class": {}
        }
    
    try:
        with open(json_path, 'r') as f:
            dataset_catalog = json.load(f)
        return dataset_catalog
    except Exception as e:
        print(f"Error loading dataset catalog: {str(e)}")
        return {
            "binary": {},
            "multi-class": {}
        }


def get_dataset_ids(
    category: Optional[str] = None,
    classes: Optional[int] = None,
    json_path: Optional[str] = None
) -> List[int]:
    """
    Get OpenML dataset IDs based on category and number of classes.
    
    Args:
        category: Dataset category ('binary' or 'multi-class').
            If None, all categories are included.
        classes: Number of classes. If None, all class counts are included.
        json_path: Path to the JSON file containing dataset information.
        
    Returns:
        List of OpenML dataset IDs matching the criteria
    """
    catalog = load_dataset_catalog(json_path)
    
    dataset_ids: List[int] = []
    
    # Filter by category
    categories = [category] if category else list(catalog.keys())
    for cat in categories:
        if cat in catalog:
            # Filter by number of classes
            for dataset_name, dataset_info in catalog[cat].items():
                # Handle both old format (value is ID) and new format (value is dict with ID)
                if isinstance(dataset_info, dict) and 'id' in dataset_info:
                    dataset_id = dataset_info['id']
                    dataset_classes = dataset_info.get('classes')
                else:
                    dataset_id = dataset_info
                    dataset_classes = None
                
                if classes is None or dataset_classes == classes:
                    dataset_ids.append(int(dataset_id))
    
    return dataset_ids


def get_recommended_datasets(
    task: str = 'classification',
    n_datasets: int = 5,
    json_path: Optional[str] = None
) -> List[int]:
    """
    Get a recommended set of datasets for building a knowledge base.
    
    Args:
        task: The ML task ('classification' or 'regression')
        n_datasets: Number of datasets to recommend
        json_path: Path to the JSON file containing dataset information
        
    Returns:
        List of recommended OpenML dataset IDs
    """
    if task == 'classification':
        # For classification, mix of binary and multi-class datasets
        binary_ids = get_dataset_ids(category='binary', json_path=json_path)
        multiclass_ids = get_dataset_ids(category='multi-class', json_path=json_path)
        
        # Prioritize some well-known datasets (iris, credit-g, etc.)
        common_datasets = [31, 61]  # credit-g, iris
        
        # Mix a balanced selection
        recommended = [id for id in common_datasets if id in binary_ids or id in multiclass_ids]
        
        # Fill remaining slots with a mix of binary and multi-class
        binary_remaining = [id for id in binary_ids if id not in recommended]
        multi_remaining = [id for id in multiclass_ids if id not in recommended]
        
        # Try to maintain a 2:1 ratio of binary to multi-class
        binary_count = max(1, int(2 * (n_datasets - len(recommended)) / 3))
        multi_count = n_datasets - len(recommended) - binary_count
        
        recommended.extend(binary_remaining[:binary_count])
        recommended.extend(multi_remaining[:multi_count])
        
        return recommended[:n_datasets]
    else:
        # For regression, we don't have categories in the current JSON
        # This could be extended once regression datasets are added to the catalog
        return [42, 505, 529, 573, 574][:n_datasets]  # Some default regression datasets 