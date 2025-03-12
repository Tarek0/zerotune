"""ZeroTune: A module for one-shot hyperparameter optimization."""

from .zerotune import (
    # Dataset meta-features calculation
    calculate_dataset_meta_parameters,
    
    # Hyperparameter transformation
    relative2absolute_dict,
    generate_random_params,
    
    # Model evaluation
    evaluate_model,
    
    # HPO with Optuna
    optuna_objective,
    optuna_hpo,
    
    # Random hyperparameter evaluation
    random_hyperparameter_evaluation,
    
    # ZeroTune training and prediction
    train_zerotune_model,
    predict_hyperparameters,
    remove_param_prefix,
)

# Knowledge Base for offline training
from .knowledge_base import KnowledgeBase

# Pre-trained predictors for inference
from .predictors import (
    get_available_models,
    ZeroTunePredictor,
    CustomZeroTunePredictor
)

__all__ = [
    # Core functionality
    'calculate_dataset_meta_parameters',
    'relative2absolute_dict',
    'generate_random_params',
    'evaluate_model',
    'optuna_objective',
    'optuna_hpo',
    'random_hyperparameter_evaluation',
    'train_zerotune_model',
    'predict_hyperparameters',
    'remove_param_prefix',
    
    # Knowledge Base
    'KnowledgeBase',
    
    # Predictors
    'get_available_models',
    'ZeroTunePredictor',
    'CustomZeroTunePredictor',
] 