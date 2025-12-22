"""
Knowledge Base retrieval for similar datasets (RAG component).

Implements similarity search to find datasets from the KB that are most
similar to the current dataset being optimized.
"""

import json
import numpy as np
from typing import Optional


# Meta-features to use for similarity computation
SIMILARITY_FEATURES = [
    'n_samples',
    'n_features', 
    'imbalance_ratio',
    'n_highly_target_corr',
    'avg_target_corr',
]


def normalize_meta_features(meta_features: dict, reference_stats: Optional[dict] = None) -> np.ndarray:
    """
    Normalize meta-features to [0, 1] range for similarity computation.
    
    Args:
        meta_features: Dictionary of meta-feature values
        reference_stats: Optional pre-computed min/max stats for normalization
        
    Returns:
        Normalized feature vector as numpy array
    """
    # Default normalization ranges based on typical dataset characteristics
    default_ranges = {
        'n_samples': (10, 100000),
        'n_features': (1, 1000),
        'imbalance_ratio': (1, 100),
        'n_highly_target_corr': (0, 100),
        'avg_target_corr': (0, 1),
        'var_target_corr': (0, 0.5),
    }
    
    if reference_stats:
        ranges = reference_stats
    else:
        ranges = default_ranges
    
    normalized = []
    for feature in SIMILARITY_FEATURES:
        value = meta_features.get(feature, 0)
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if max_val > min_val:
                norm_value = (value - min_val) / (max_val - min_val)
                norm_value = max(0, min(1, norm_value))  # Clip to [0, 1]
            else:
                norm_value = 0.5
        else:
            norm_value = value
        normalized.append(norm_value)
    
    return np.array(normalized)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def retrieve_similar_datasets(
    query_meta_features: dict,
    kb_path: str,
    n_similar: int = 3,
    exclude_dataset: Optional[str] = None
) -> list[dict]:
    """
    Retrieve similar datasets from knowledge base using cosine similarity.
    
    Args:
        query_meta_features: Meta-features of the dataset to find similar examples for
        kb_path: Path to the knowledge base JSON file
        n_similar: Number of similar datasets to retrieve
        exclude_dataset: Optional dataset name to exclude (e.g., the query dataset itself)
        
    Returns:
        List of similar dataset info dictionaries with:
        - name: Dataset name
        - similarity: Similarity score
        - meta_features: Dataset meta-features
        - best_params: Best hyperparameters found for this dataset
        - best_score: Best score achieved
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    
    datasets = kb.get('datasets', [])
    results = kb.get('results', [])
    
    # Build mapping from dataset name to best result
    best_results = {}
    for result in results:
        name = result.get('dataset_name', '')
        score = result.get('best_score', 0)
        
        if name not in best_results or score > best_results[name].get('best_score', 0):
            best_results[name] = result
    
    # Normalize query features
    query_vec = normalize_meta_features(query_meta_features)
    
    # Compute similarities
    similarities = []
    for dataset in datasets:
        name = dataset.get('name', '')
        
        if exclude_dataset and name == exclude_dataset:
            continue
        
        meta = dataset.get('meta_features', {})
        dataset_vec = normalize_meta_features(meta)
        
        sim = cosine_similarity(query_vec, dataset_vec)
        
        # Get best result for this dataset
        result = best_results.get(name, {})
        
        similarities.append({
            'name': name,
            'similarity': sim,
            'meta_features': meta,
            'best_params': result.get('best_hyperparameters', {}),
            'best_score': result.get('best_score', None),
            'dataset_id': dataset.get('id'),
        })
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:n_similar]


def format_similar_datasets_for_prompt(similar_datasets: list[dict]) -> str:
    """
    Format retrieved similar datasets for inclusion in LLM prompt.
    
    Args:
        similar_datasets: List of similar dataset dictionaries from retrieve_similar_datasets
        
    Returns:
        Formatted string for prompt
    """
    if not similar_datasets:
        return "No similar datasets found in knowledge base."
    
    lines = []
    for i, ds in enumerate(similar_datasets, 1):
        meta = ds.get('meta_features', {})
        params = ds.get('best_params', {})
        score = ds.get('best_score')
        
        # Format imbalance ratio safely
        imb = meta.get('imbalance_ratio')
        imb_str = f"{imb:.2f}" if isinstance(imb, (int, float)) else "N/A"
        
        # Format score safely
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        
        lines.append(f"""
### Similar Dataset {i}: {ds['name']} (similarity: {ds['similarity']:.3f})
- Samples: {meta.get('n_samples', 'N/A')}, Features: {meta.get('n_features', 'N/A')}
- Imbalance ratio: {imb_str}
- Best hyperparameters found: {_format_params(params)}
- Best score achieved: {score_str}""")
    
    return "\n".join(lines)


def _format_params(params: dict) -> str:
    """Format parameters dictionary for display."""
    if not params:
        return "N/A"
    
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    
    return ", ".join(parts)
