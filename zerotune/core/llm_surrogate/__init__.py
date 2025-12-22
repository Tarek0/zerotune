"""
LLM Surrogate module for ZeroTune.

This module implements an LLM-based surrogate model for hyperparameter optimization,
designed as a custom Optuna Sampler that can replace traditional methods like TPE.

Usage:
    from zerotune.core.llm_surrogate import LLMSurrogateSampler
    from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
    
    # Calculate meta-features
    meta_features = calculate_dataset_meta_parameters(X, y)
    
    # Create sampler
    sampler = LLMSurrogateSampler(
        meta_features=meta_features,
        kb_path="knowledge_base/kb_decision_tree.json",
        model_type="decision_tree",
        llm_model="gpt-5-mini-2025-08-07"
    )
    
    # Use with Optuna
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)
"""

from zerotune.core.llm_surrogate.schemas import (
    TrialAnalysis,
    HPORecommendation,
    InitialRecommendation,
)
from zerotune.core.llm_surrogate.sampler import LLMSurrogateSampler
from zerotune.core.llm_surrogate.prompt_builder import PromptBuilder
from zerotune.core.llm_surrogate.param_guides import get_param_guide
from zerotune.core.llm_surrogate.kb_stats import compute_kb_statistics
from zerotune.core.llm_surrogate.retrieval import (
    retrieve_similar_datasets,
    format_similar_datasets_for_prompt,
)

__all__ = [
    "LLMSurrogateSampler",
    "PromptBuilder",
    "TrialAnalysis",
    "HPORecommendation",
    "InitialRecommendation",
    "get_param_guide",
    "compute_kb_statistics",
    "retrieve_similar_datasets",
    "format_similar_datasets_for_prompt",
]
