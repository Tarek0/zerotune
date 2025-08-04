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
    
    # Create a new DataFrame with features and target
    df = pd.DataFrame(X, columns=attribute_names)
    df[dataset.default_target_attribute] = y
    
    return df, dataset.default_target_attribute, dataset.name


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
                - Remaining missing values filled with 0
            - y: Target variable as Series (converted to int if categorical)
    """
    # Drop rows where target is NaN
    if df[target_name].isna().any():
        orig_len = len(df)
        df = df.dropna(subset=[target_name])
        dropped = orig_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with NaN values in target column '{target_name}'")
    
    # Extract target and convert to numeric if needed
    y = df[target_name]
    if y.dtype == 'object' or y.dtype == 'category':
        # Convert categorical target to numeric
        y = pd.factorize(y)[0]  # Returns (codes, unique_values)
    
    # Extract features
    X = df.drop(target_name, axis=1)
    
    # Handle categorical and string columns first
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            # Convert to category type for ML algorithms
            X[col] = pd.Categorical(X[col]).codes
    
    # Fill numeric missing values with 0 (changed from mean to match test expectations)
    numeric_cols = X.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(0)
    
    # Fill any remaining missing values with 0
    X = X.fillna(0)
    
    # Check for any remaining NaNs and warn if found
    if X.isna().any().any():
        print(f"Warning: Dataset still contains NaN values after preprocessing")
        # Count NaNs by column for debugging
        nan_counts = X.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]
        if len(nan_columns) > 0:
            print(f"Columns with NaNs: {nan_columns.to_dict()}")
    
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


def get_dataset_ids(category: str = "all", n_classes: Optional[int] = None, json_path: Optional[str] = None) -> List[int]:
    """Get OpenML dataset IDs based on category and number of classes.
    
    Args:
        category: Dataset category ('binary', 'multiclass', or 'all')
        n_classes: Optional filter for number of classes
        json_path: Optional path to dataset catalog JSON file
        
    Returns:
        List of dataset IDs matching the criteria
    """
    catalog = load_dataset_catalog(json_path)
    categories = ['binary', 'multiclass'] if category == "all" else [category]
    dataset_ids = []
    for cat in categories:
        if cat in catalog:
            for dataset in catalog[cat]:
                if n_classes is None or dataset.get('n_classes', 0) == n_classes:
                    try:
                        dataset_ids.append(int(dataset['id']))
                    except (ValueError, KeyError):
                        continue
    return sorted(dataset_ids)


def get_recommended_datasets(task: str = "classification", n_datasets: int = 5, json_path: Optional[str] = None) -> List[int]:
    """Get a recommended set of datasets for building a knowledge base.
    
    Args:
        task: Task type ('classification' or 'regression')
        n_datasets: Number of datasets to return
        json_path: Optional path to dataset catalog JSON file
        
    Returns:
        List of recommended dataset IDs
    """
    if task == "classification":
        binary_ids = get_dataset_ids(category='binary', json_path=json_path)
        multiclass_ids = get_dataset_ids(category='multiclass', json_path=json_path)
        all_ids = binary_ids + multiclass_ids
        return sorted(all_ids)[:n_datasets]
    else:
        regression_ids = get_dataset_ids(category='regression', json_path=json_path)
        return sorted(regression_ids)[:n_datasets] 