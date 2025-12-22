"""
Prompt Builder for LLM Surrogate HPO.

Constructs layered prompts that combine:
1. Static expert knowledge (parameter guides)
2. Empirical knowledge (KB statistics, similar datasets)
3. Current context (meta-features, search space, trial history)
"""

from typing import Optional
import optuna

from zerotune.core.llm_surrogate.param_guides import get_param_guide
from zerotune.core.llm_surrogate.kb_stats import compute_kb_statistics
from zerotune.core.llm_surrogate.retrieval import (
    retrieve_similar_datasets,
    format_similar_datasets_for_prompt,
)


INITIAL_PROMPT_TEMPLATE = """You are an expert ML practitioner optimizing hyperparameters for a {model_class}.

{param_guide}

{kb_statistics}

## Current Dataset Characteristics
{meta_features_formatted}

## Search Space
{search_space_formatted}

## Similar Datasets from Knowledge Base
{similar_datasets_formatted}

## Task
Based on all the above context, recommend the FIRST set of hyperparameters to try.

Consider:
1. The dataset size category and appropriate complexity level
2. Risk of overfitting given the feature/sample ratio
3. Class imbalance implications
4. Patterns from similar datasets in the knowledge base
5. The parameter interactions and heuristics from the guide

Provide step-by-step reasoning, then your recommended hyperparameters."""


SEQUENTIAL_PROMPT_TEMPLATE = """You are an expert ML practitioner optimizing hyperparameters for a {model_class}.

{param_guide}

{kb_statistics}

## Current Dataset Characteristics
{meta_features_formatted}

## Search Space
{search_space_formatted}

## Optimization History ({n_completed_trials} trials completed)
Best score so far: {best_score:.4f}

### Trial History (sorted by score, best first)
{trial_history_formatted}

## Similar Datasets from Knowledge Base
{similar_datasets_formatted}

## Task
Based on the observed trial results and expert knowledge above, recommend FOUR diverse configurations:

1. **EXPLOIT**: Small refinements to the best configuration (adjust 1-2 params slightly)
2. **EXPLORE_MILD**: A nearby but unexplored region (change 1-2 params more significantly)
3. **EXPLORE_AGGRESSIVE**: A very different region (opposite end of key parameters)
4. **WILDCARD**: High-risk, high-reward (extreme or unusual combinations)

Analyze:
1. Which hyperparameter changes correlated with score improvements?
2. Are there signs of overfitting (complex models performing worse)?
3. Are there signs of underfitting (simple models plateaued)?
4. Based on parameter interactions in the guide, what adjustments make sense?

Guidelines:
- EXPLOIT: Stay close to best (e.g., if best has max_depth=8, try 7 or 9)
- EXPLORE_MILD: Moderate changes (e.g., if best has max_depth=8, try 5 or 12)
- EXPLORE_AGGRESSIVE: Very different (e.g., if best is deep+simple, try shallow+complex)
- WILDCARD: Edge cases or unusual combos that might surprise (e.g., extreme regularization)

Provide all four recommendations with brief reasoning for each."""


class PromptBuilder:
    """Builds layered prompts with expert knowledge and KB context."""
    
    def __init__(
        self,
        model_type: str,
        kb_path: str,
        n_similar_datasets: int = 3
    ):
        """
        Initialize the prompt builder.
        
        Args:
            model_type: Type of model ('decision_tree', 'random_forest', 'xgboost')
            kb_path: Path to the knowledge base JSON file
            n_similar_datasets: Number of similar datasets to retrieve for context
        """
        self.model_type = model_type
        self.kb_path = kb_path
        self.n_similar_datasets = n_similar_datasets
        
        # Load static expert knowledge
        self.param_guide = get_param_guide(model_type)
        
        # Compute KB statistics (cached)
        self.kb_statistics = compute_kb_statistics(kb_path, model_type)
        
        # Model class name mapping
        self._model_class_names = {
            'decision_tree': 'DecisionTreeClassifier',
            'random_forest': 'RandomForestClassifier',
            'xgboost': 'XGBoostClassifier',
        }
    
    def build_initial_prompt(
        self,
        meta_features: dict,
        search_space: dict,
    ) -> str:
        """
        Build prompt for initial (zero-shot) recommendation.
        
        Args:
            meta_features: Dataset meta-features dictionary
            search_space: Optuna search space or parameter grid
            
        Returns:
            Formatted prompt string
        """
        # Retrieve similar datasets
        similar = retrieve_similar_datasets(
            meta_features,
            self.kb_path,
            n_similar=self.n_similar_datasets
        )
        
        model_class = self._model_class_names.get(self.model_type, self.model_type)
        meta_str = self._format_meta_features(meta_features)
        search_str = self._format_search_space(search_space)
        similar_str = format_similar_datasets_for_prompt(similar)
        
        # Build prompt by concatenation to avoid format() issues with curly braces
        prompt = f"""You are an expert ML practitioner optimizing hyperparameters for a {model_class}.

{self.param_guide}

{self.kb_statistics}

## Current Dataset Characteristics
{meta_str}

## Search Space
{search_str}

## Similar Datasets from Knowledge Base
{similar_str}

## Task
Based on all the above context, recommend the FIRST set of hyperparameters to try.

Consider:
1. The dataset size category and appropriate complexity level
2. Risk of overfitting given the feature/sample ratio
3. Class imbalance implications
4. Patterns from similar datasets in the knowledge base
5. The parameter interactions and heuristics from the guide

Provide step-by-step reasoning, then your recommended hyperparameters."""
        
        return prompt
    
    def build_sequential_prompt(
        self,
        meta_features: dict,
        search_space: dict,
        completed_trials: list,
    ) -> str:
        """
        Build prompt for sequential recommendation based on trial history.
        
        Args:
            meta_features: Dataset meta-features dictionary
            search_space: Optuna search space or parameter grid
            completed_trials: List of completed Optuna trials
            
        Returns:
            Formatted prompt string
        """
        # Retrieve similar datasets
        similar = retrieve_similar_datasets(
            meta_features,
            self.kb_path,
            n_similar=self.n_similar_datasets
        )
        
        # Find best score
        best_score = max(t.value for t in completed_trials) if completed_trials else 0.0
        
        model_class = self._model_class_names.get(self.model_type, self.model_type)
        meta_str = self._format_meta_features(meta_features)
        search_str = self._format_search_space(search_space)
        trial_str = self._format_trial_history(completed_trials)
        similar_str = format_similar_datasets_for_prompt(similar)
        n_trials = len(completed_trials)
        
        # Build prompt by concatenation to avoid format() issues
        prompt = f"""You are an expert ML practitioner optimizing hyperparameters for a {model_class}.

{self.param_guide}

{self.kb_statistics}

## Current Dataset Characteristics
{meta_str}

## Search Space
{search_str}

## Optimization History ({n_trials} trials completed)
Best score so far: {best_score:.4f}

### Trial History (sorted by score, best first)
{trial_str}

## Similar Datasets from Knowledge Base
{similar_str}

## Task
Based on the observed trial results and expert knowledge above, recommend FOUR diverse configurations:

1. **EXPLOIT**: Small refinements to the best configuration (adjust 1-2 params slightly)
2. **EXPLORE_MILD**: A nearby but unexplored region (change 1-2 params more significantly)
3. **EXPLORE_AGGRESSIVE**: A very different region (opposite end of key parameters)
4. **WILDCARD**: High-risk, high-reward (extreme or unusual combinations)

Analyze:
1. Which hyperparameter changes correlated with score improvements?
2. Are there signs of overfitting (complex models performing worse)?
3. Are there signs of underfitting (simple models plateaued)?
4. Based on parameter interactions in the guide, what adjustments make sense?

Guidelines:
- EXPLOIT: Stay close to best (e.g., if best has max_depth=8, try 7 or 9)
- EXPLORE_MILD: Moderate changes (e.g., if best has max_depth=8, try 5 or 12)
- EXPLORE_AGGRESSIVE: Very different (e.g., if best is deep+simple, try shallow+complex)
- WILDCARD: Edge cases or unusual combos that might surprise (e.g., extreme regularization)

Provide all four recommendations with brief reasoning for each."""
        
        return prompt
    
    def _format_meta_features(self, meta_features: dict) -> str:
        """Format meta-features for prompt."""
        n_samples = meta_features.get('n_samples', 0)
        n_features = meta_features.get('n_features', 0)
        
        # Compute derived metrics
        feature_sample_ratio = n_features / n_samples if n_samples > 0 else 0
        
        # Determine size category
        if n_samples < 1000:
            size_category = "small (< 1000 samples)"
        elif n_samples < 10000:
            size_category = "medium (1000-10000 samples)"
        else:
            size_category = "large (>= 10000 samples)"
        
        lines = [
            f"- Samples: {n_samples:,}",
            f"- Features: {n_features}",
            f"- Size category: {size_category}",
            f"- Feature/Sample ratio: {feature_sample_ratio:.4f}",
            f"- Class imbalance ratio: {meta_features.get('imbalance_ratio', 1):.2f}",
            f"- Features highly correlated with target: {meta_features.get('n_highly_target_corr', 0)}",
            f"- Average target correlation: {meta_features.get('avg_target_corr', 0):.4f}",
        ]
        
        return "\n".join(lines)
    
    def _format_search_space(self, search_space: dict) -> str:
        """Format search space for prompt."""
        lines = []
        
        for param, spec in search_space.items():
            if isinstance(spec, list):
                # Categorical
                lines.append(f"- {param}: categorical {spec}")
            elif isinstance(spec, tuple) and len(spec) == 3:
                low, high, scale = spec
                if scale == 'int':
                    lines.append(f"- {param}: integer [{int(low)}, {int(high)}]")
                elif scale == 'log':
                    lines.append(f"- {param}: float [{low}, {high}] (log scale)")
                else:
                    lines.append(f"- {param}: float [{low}, {high}]")
            elif hasattr(spec, 'low') and hasattr(spec, 'high'):
                # Optuna distribution
                lines.append(f"- {param}: [{spec.low}, {spec.high}]")
            else:
                lines.append(f"- {param}: {spec}")
        
        return "\n".join(lines)
    
    def _format_trial_history(self, trials: list, max_trials: int = 15) -> str:
        """Format Optuna trials for prompt."""
        if not trials:
            return "No trials completed yet."
        
        # Sort by score (descending)
        sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
        
        lines = []
        for t in sorted_trials[:max_trials]:
            params_str = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(t.params.items())
            )
            lines.append(f"Trial {t.number}: score={t.value:.4f} | {params_str}")
        
        if len(sorted_trials) > max_trials:
            lines.append(f"... and {len(sorted_trials) - max_trials} more trials")
        
        return "\n".join(lines)
