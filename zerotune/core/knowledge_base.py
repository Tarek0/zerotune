"""
Knowledge Base management functions for ZeroTune.

This module handles operations related to the knowledge base, including
creating, loading, saving, and updating knowledge bases that store
information about datasets and their optimal hyperparameter configurations.
"""

import os
import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from zerotune.core.config import CONFIG, DEFAULT_KB_DIR, get_knowledge_base_path
from zerotune.core.utils import safe_json_serialize, load_json, save_json

# Type aliases
KnowledgeBase = Dict[str, Any]
MetaFeatures = Dict[str, float]
HyperParams = Dict[str, Any]
OptimizationResults = List[Dict[str, Any]]


def initialize_knowledge_base() -> KnowledgeBase:
    """
    Initialize an empty knowledge base structure.
    
    Returns:
        Empty knowledge base with proper structure
    """
    return {
        "meta_features": [],
        "results": [],
        "datasets": [],
        "created_at": time.time(),
        "updated_at": time.time()
    }


def load_knowledge_base(file_path: str, model_type: Optional[str] = None) -> KnowledgeBase:
    """
    Load a knowledge base from a JSON file.
    
    Args:
        file_path: Path to the knowledge base file
        model_type: Model type to load data for. If specified,
                   loads only data for this model type.
    
    Returns:
        Loaded knowledge base or an empty one if loading fails
    """
    # If model type is specified, modify the path to use model-specific file
    model_specific_file = False
    model_file_path = None
    
    if model_type:
        # Get directory and base filename without extension
        dir_path = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Construct new filename with model type
        model_specific_filename = f"{base_name}_{model_type}.json"
        model_file_path = os.path.join(dir_path, model_specific_filename)
        
        # Use model-specific file if it exists
        if os.path.exists(model_file_path):
            file_path = model_file_path
            model_specific_file = True
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Load the knowledge base
    kb = load_json(file_path, default=None)
    
    if kb is not None:
        # If we're using a regular file but model_type is specified,
        # filter the results to include only that model type
        if model_type and not model_specific_file and "results" in kb:
            kb["results"] = [r for r in kb["results"] 
                             if r.get("model_type") == model_type]
        return kb
    else:
        # Create an empty knowledge base
        kb = initialize_knowledge_base()
        # Save the new empty knowledge base
        save_json(kb, file_path)
        return kb


def save_knowledge_base(
    knowledge_base: KnowledgeBase,
    file_path: str,
    model_type: Optional[str] = None
) -> bool:
    """
    Save a knowledge base to a JSON file.
    
    Args:
        knowledge_base: Knowledge base to save
        file_path: Path to save the knowledge base to
        model_type: Model type to save data for. If specified,
                   saves to a model-specific file.
    
    Returns:
        True if successful, False otherwise
    """
    # If model type is specified, modify the path to use model-specific file
    if model_type:
        # Get directory and base filename without extension
        dir_path = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Construct new filename with model type
        model_specific_filename = f"{base_name}_{model_type}.json"
        file_path = os.path.join(dir_path, model_specific_filename)
    
    # Update the timestamp
    knowledge_base["updated_at"] = time.time()
    
    # Save the knowledge base using the util function
    return save_json(knowledge_base, file_path)


def update_knowledge_base(
    knowledge_base: KnowledgeBase,
    dataset_name: str,
    meta_features: MetaFeatures,
    best_hyperparameters: HyperParams,
    best_score: float,
    all_results: Optional[OptimizationResults] = None,
    dataset_id: Optional[int] = None,
    model_type: Optional[str] = None
) -> KnowledgeBase:
    """
    Update knowledge base with results from a new dataset.
    
    Args:
        knowledge_base: Knowledge base to update
        dataset_name: Name of the dataset
        meta_features: Meta-features of the dataset
        best_hyperparameters: Best hyperparameters found
        best_score: Best score achieved
        all_results: All hyperparameter configurations tried
        dataset_id: ID of the dataset (e.g., OpenML dataset ID)
        model_type: Type of model used (e.g., 'decision_tree')
        
    Returns:
        Updated knowledge base
    """
    # Update meta features - check if dataset already exists to avoid duplication
    if "meta_features" not in knowledge_base:
        knowledge_base["meta_features"] = []
    
    # Check if we already have this dataset in the knowledge base
    dataset_in_kb = False
    if "datasets" in knowledge_base:
        for dataset in knowledge_base["datasets"]:
            if dataset.get("name") == dataset_name:
                dataset_in_kb = True
                break
    
    # Only add meta-features if this is a new dataset
    if not dataset_in_kb:
        knowledge_base["meta_features"].append(meta_features)
    
    # Update results
    if "results" not in knowledge_base:
        knowledge_base["results"] = []
    
    # Create the result entry
    result_entry = {
        "dataset_name": dataset_name,
        "best_hyperparameters": best_hyperparameters,
        "best_score": best_score,
        "timestamp": time.time()
    }
    
    if dataset_id:
        result_entry["dataset_id"] = dataset_id
    
    if model_type:
        result_entry["model_type"] = model_type
    
    if all_results:
        result_entry["all_results"] = all_results
    
    # Add the result
    knowledge_base["results"].append(result_entry)
    
    # Update datasets
    if "datasets" not in knowledge_base:
        knowledge_base["datasets"] = []
    
    # Check if dataset exists already in the datasets list
    dataset_exists = False
    for i, dataset in enumerate(knowledge_base["datasets"]):
        if dataset.get("name") == dataset_name:
            # Update existing dataset entry
            dataset_exists = True
            knowledge_base["datasets"][i]["updated_at"] = time.time()
            
            # Update meta features if provided
            if "meta_features" in knowledge_base["datasets"][i]:
                knowledge_base["datasets"][i]["meta_features"].update(meta_features)
            else:
                knowledge_base["datasets"][i]["meta_features"] = meta_features
                
            # Update dataset ID if provided
            if dataset_id:
                knowledge_base["datasets"][i]["id"] = dataset_id
            
            break
    
    # Add new dataset if it doesn't exist
    if not dataset_exists:
        dataset_entry = {
            "name": dataset_name,
            "meta_features": meta_features,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        if dataset_id:
            dataset_entry["id"] = dataset_id
        
        knowledge_base["datasets"].append(dataset_entry)
    
    # Update the knowledge base's update timestamp
    knowledge_base["updated_at"] = time.time()
    
    return knowledge_base


# Function now uses the config module to determine path
def get_knowledge_base_path() -> str:
    """
    Get the path for the knowledge base file.
    
    Returns:
        Path to the knowledge base file
    """
    # Use the function from the config module
    return get_knowledge_base_path("default") 