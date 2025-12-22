"""
Knowledge Base statistics computation for LLM prompts.

Computes aggregate statistics from historical HPO experiments to provide
empirical grounding for LLM recommendations.
"""

import json
import numpy as np
from collections import defaultdict
from typing import Optional
from pathlib import Path


def compute_kb_statistics(kb_path: str, model_type: Optional[str] = None) -> str:
    """
    Compute aggregate statistics from knowledge base.
    
    Args:
        kb_path: Path to the knowledge base JSON file
        model_type: Optional model type to filter results
        
    Returns:
        Formatted string with KB statistics for prompt inclusion
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    
    results = kb.get('results', [])
    datasets = kb.get('datasets', [])
    
    if model_type:
        results = [r for r in results if r.get('model_type', '').lower() == model_type.lower()]
    
    if not results:
        return "## Knowledge Base Statistics\nNo historical data available for this model type."
    
    # Extract optimal parameters from all datasets
    optimal_params = defaultdict(list)
    scores = []
    
    for result in results:
        best_params = result.get('best_hyperparameters', {})
        for param, value in best_params.items():
            if value is not None and isinstance(value, (int, float)):
                optimal_params[param].append(value)
        
        if 'best_score' in result:
            scores.append(result['best_score'])
    
    # Build statistics text
    stats_lines = [
        f"## Historical Patterns from Knowledge Base ({len(datasets)} datasets, {len(results)} experiments)",
        "",
        "### Optimal Parameter Distributions",
    ]
    
    for param, values in sorted(optimal_params.items()):
        if not values:
            continue
            
        values_arr = np.array(values)
        stats_lines.append(f"""
**{param}**:
- Range: {np.min(values_arr):.3f} - {np.max(values_arr):.3f}
- Median: {np.median(values_arr):.3f}
- Mean: {np.mean(values_arr):.3f}
- Std: {np.std(values_arr):.3f}""")
    
    if scores:
        scores_arr = np.array(scores)
        stats_lines.append(f"""
### Performance Distribution
- Best score range: {np.min(scores_arr):.4f} - {np.max(scores_arr):.4f}
- Median score: {np.median(scores_arr):.4f}
- Mean score: {np.mean(scores_arr):.4f}""")
    
    return "\n".join(stats_lines)


def compute_param_correlations(kb_path: str, model_type: Optional[str] = None) -> dict:
    """
    Compute correlations between meta-features and optimal parameters.
    
    Args:
        kb_path: Path to the knowledge base JSON file
        model_type: Optional model type to filter results
        
    Returns:
        Dictionary mapping (meta_feature, param) pairs to correlation values
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    
    results = kb.get('results', [])
    datasets = kb.get('datasets', [])
    
    if model_type:
        results = [r for r in results if r.get('model_type', '').lower() == model_type.lower()]
    
    # Build mapping from dataset name to meta-features
    dataset_meta = {}
    for d in datasets:
        dataset_meta[d['name']] = d.get('meta_features', {})
    
    # Collect data for correlation analysis
    meta_feature_names = set()
    param_names = set()
    data_rows = []
    
    for result in results:
        dataset_name = result.get('dataset_name', '')
        if dataset_name not in dataset_meta:
            continue
            
        meta = dataset_meta[dataset_name]
        params = result.get('best_hyperparameters', {})
        
        meta_feature_names.update(meta.keys())
        param_names.update(params.keys())
        
        data_rows.append({
            'meta': meta,
            'params': params
        })
    
    if len(data_rows) < 3:
        return {}
    
    # Compute correlations
    correlations = {}
    
    for mf_name in meta_feature_names:
        mf_values = []
        for row in data_rows:
            val = row['meta'].get(mf_name)
            if val is not None and isinstance(val, (int, float)):
                mf_values.append(val)
            else:
                mf_values.append(np.nan)
        
        mf_arr = np.array(mf_values)
        
        for p_name in param_names:
            p_values = []
            for row in data_rows:
                val = row['params'].get(p_name)
                if val is not None and isinstance(val, (int, float)):
                    p_values.append(val)
                else:
                    p_values.append(np.nan)
            
            p_arr = np.array(p_values)
            
            # Compute correlation ignoring NaN
            valid_mask = ~(np.isnan(mf_arr) | np.isnan(p_arr))
            if np.sum(valid_mask) >= 3:
                corr = np.corrcoef(mf_arr[valid_mask], p_arr[valid_mask])[0, 1]
                if not np.isnan(corr):
                    correlations[(mf_name, p_name)] = corr
    
    return correlations


def get_dataset_size_category(n_samples: int) -> str:
    """Categorize dataset by size."""
    if n_samples < 1000:
        return "small"
    elif n_samples < 10000:
        return "medium"
    else:
        return "large"


def compute_conditional_statistics(
    kb_path: str, 
    model_type: Optional[str] = None
) -> dict:
    """
    Compute statistics conditioned on dataset characteristics.
    
    Returns statistics like "for small datasets, optimal max_depth is typically 4-8".
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    
    results = kb.get('results', [])
    datasets = kb.get('datasets', [])
    
    if model_type:
        results = [r for r in results if r.get('model_type', '').lower() == model_type.lower()]
    
    # Build mapping from dataset name to meta-features
    dataset_meta = {}
    for d in datasets:
        dataset_meta[d['name']] = d.get('meta_features', {})
    
    # Group by dataset size category
    by_size = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        dataset_name = result.get('dataset_name', '')
        if dataset_name not in dataset_meta:
            continue
        
        meta = dataset_meta[dataset_name]
        n_samples = meta.get('n_samples', 0)
        size_cat = get_dataset_size_category(n_samples)
        
        params = result.get('best_hyperparameters', {})
        for param, value in params.items():
            if value is not None and isinstance(value, (int, float)):
                by_size[size_cat][param].append(value)
    
    # Compute statistics per category
    conditional_stats = {}
    for size_cat, param_dict in by_size.items():
        conditional_stats[size_cat] = {}
        for param, values in param_dict.items():
            if values:
                conditional_stats[size_cat][param] = {
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
    
    return conditional_stats
