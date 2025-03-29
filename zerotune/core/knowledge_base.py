"""
Knowledge Base management functions for ZeroTune.

This module handles operations related to the knowledge base, including
creating, loading, saving, and updating knowledge bases that store
information about datasets and their optimal hyperparameter configurations.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

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
    
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check if file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                kb = json.load(f)
                
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
            with open(file_path, 'w') as f:
                json.dump(kb, f, indent=2)
            return kb
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return initialize_knowledge_base()


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
    
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Update the timestamp
        knowledge_base["updated_at"] = time.time()
        
        # Convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
        
        # Convert numpy types in the knowledge base
        kb_to_save = convert_numpy_types(knowledge_base)
        
        # Save the knowledge base
        with open(file_path, 'w') as f:
            json.dump(kb_to_save, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving knowledge base: {e}")
        return False


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
    
    result = {
        "dataset_name": dataset_name,
        "best_hyperparameters": best_hyperparameters,
        "best_score": best_score
    }
    
    if dataset_id is not None:
        result["dataset_id"] = dataset_id
    
    if all_results is not None:
        result["all_results"] = all_results
    
    if model_type is not None:
        result["model_type"] = model_type
    
    # Check if we already have a result for this dataset and model type
    existing_result_index = None
    for i, r in enumerate(knowledge_base["results"]):
        if (r.get("dataset_name") == dataset_name and 
            r.get("model_type") == model_type):
            existing_result_index = i
            break
    
    # Update existing result or add new one
    if existing_result_index is not None:
        knowledge_base["results"][existing_result_index] = result
    else:
        knowledge_base["results"].append(result)
    
    # Update datasets list
    if "datasets" not in knowledge_base:
        knowledge_base["datasets"] = []
    
    # Check if dataset is already in the list
    dataset_exists = False
    for dataset in knowledge_base["datasets"]:
        if dataset.get("name") == dataset_name:
            # Update the dataset entry if needed
            if dataset_id is not None and dataset.get("id") != dataset_id:
                dataset["id"] = dataset_id
            dataset_exists = True
            break
    
    # Add new dataset entry if it doesn't exist
    if not dataset_exists:
        dataset_info = {"name": dataset_name}
        if dataset_id is not None:
            dataset_info["id"] = dataset_id
        knowledge_base["datasets"].append(dataset_info)
    
    # Update timestamp
    knowledge_base["updated_at"] = time.time()
    
    return knowledge_base


def get_knowledge_base_path(base_dir: str = "knowledge_base", filename: str = "kb.json") -> str:
    """
    Get the path to the knowledge base file.
    
    Args:
        base_dir: Base directory for knowledge base
        filename: Filename for knowledge base
        
    Returns:
        Full path to knowledge base file
    """
    # Get the directory of the current file
    current_dir = Path(__file__).parent.parent.parent
    
    # Construct the path to the knowledge base
    kb_dir = current_dir / base_dir
    
    # Create directory if it doesn't exist
    os.makedirs(kb_dir, exist_ok=True)
    
    return str(kb_dir / filename) 