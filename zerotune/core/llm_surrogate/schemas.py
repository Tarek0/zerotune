"""
Pydantic schemas for structured LLM outputs in HPO.

These schemas ensure the LLM returns properly structured hyperparameter
recommendations that can be directly used by Optuna.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class TrialAnalysis(BaseModel):
    """Analysis of a single observed trial."""
    
    trial_number: int = Field(description="The trial number being analyzed")
    observation: Optional[str] = Field(default=None, description="What was observed in this trial's results")
    implication: Optional[str] = Field(default=None, description="What this implies for the next trial")


class DecisionTreeHyperparameters(BaseModel):
    """Hyperparameters for DecisionTreeClassifier."""
    
    max_depth: int = Field(ge=1, le=20, description="Maximum tree depth (1-20)")
    min_samples_split: int = Field(ge=2, le=50, description="Min samples to split (2-50)")
    min_samples_leaf: int = Field(ge=1, le=25, description="Min samples per leaf (1-25)")
    criterion: Literal["gini", "entropy"] = Field(description="Split criterion")


class RandomForestHyperparameters(BaseModel):
    """Hyperparameters for RandomForestClassifier."""
    
    n_estimators: int = Field(ge=50, le=500, description="Number of trees (50-500)")
    max_depth: int = Field(ge=1, le=20, description="Maximum tree depth (1-20)")
    min_samples_split: int = Field(ge=2, le=20, description="Min samples to split (2-20)")
    min_samples_leaf: int = Field(ge=1, le=15, description="Min samples per leaf (1-15)")
    max_features: Literal["sqrt", "log2"] = Field(description="Features per split")


class XGBoostHyperparameters(BaseModel):
    """Hyperparameters for XGBoostClassifier."""
    
    n_estimators: int = Field(ge=50, le=500, description="Number of boosting rounds (50-500)")
    max_depth: int = Field(ge=3, le=12, description="Maximum tree depth (3-12)")
    learning_rate: float = Field(ge=0.01, le=0.3, description="Learning rate (0.01-0.3)")
    subsample: float = Field(ge=0.5, le=1.0, description="Subsample ratio (0.5-1.0)")
    colsample_bytree: float = Field(ge=0.5, le=1.0, description="Column sample ratio (0.5-1.0)")


class ConfigRecommendation(BaseModel):
    """A single hyperparameter configuration with reasoning."""
    
    reasoning: str = Field(description="Brief reasoning for this configuration")
    hyperparameters: DecisionTreeHyperparameters = Field(description="The hyperparameters")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence (0-1)")


class HPORecommendation(BaseModel):
    """Structured output with 4 diverse recommendations per LLM call."""
    
    trial_analyses: list[TrialAnalysis] = Field(
        default_factory=list,
        description="Analysis of top 2-3 trials from history"
    )
    
    overall_assessment: str = Field(
        description="Brief assessment: overfitting/underfitting/converging"
    )
    
    exploit: ConfigRecommendation = Field(
        description="EXPLOIT: Small refinements to the best configuration"
    )
    
    explore_mild: ConfigRecommendation = Field(
        description="EXPLORE_MILD: A nearby but unexplored region"
    )
    
    explore_aggressive: ConfigRecommendation = Field(
        description="EXPLORE_AGGRESSIVE: A very different region of the search space"
    )
    
    wildcard: ConfigRecommendation = Field(
        description="WILDCARD: A high-risk, high-reward configuration (e.g., extreme values)"
    )


class InitialRecommendation(BaseModel):
    """Structured output for zero-shot initial recommendation."""
    
    dataset_assessment: str = Field(
        description="Assessment of the dataset characteristics"
    )
    
    strategy: str = Field(
        description="Overall strategy (e.g., 'regularization-focused')"
    )
    
    reasoning_summary: str = Field(
        description="Concise reasoning behind this recommendation"
    )
    
    hyperparameters: DecisionTreeHyperparameters = Field(
        description="The recommended hyperparameters"
    )
    
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence in this recommendation (0-1)"
    )


def get_hyperparameter_schema(model_type: str):
    """Get the appropriate hyperparameter schema for a model type."""
    schemas = {
        'decision_tree': DecisionTreeHyperparameters,
        'random_forest': RandomForestHyperparameters,
        'xgboost': XGBoostHyperparameters,
    }
    return schemas.get(model_type.lower(), DecisionTreeHyperparameters)
