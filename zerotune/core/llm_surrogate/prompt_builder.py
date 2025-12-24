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
from zerotune.core.llm_surrogate.kb_stats import (
    compute_conditional_statistics,
    compute_param_correlations,
    get_dataset_size_category,
)
from zerotune.core.llm_surrogate.retrieval import (
    retrieve_similar_datasets,
    format_similar_datasets_for_prompt,
)


class PromptBuilder:
    """Builds layered prompts with expert knowledge and KB context."""
    
    def __init__(
        self,
        model_type: str,
        kb_path: str,
        n_similar_datasets: int = 3,
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
        self.conditional_stats = compute_conditional_statistics(kb_path, model_type)
        self.correlations = compute_param_correlations(kb_path, model_type)
        
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
        zt_prior: Optional[dict] = None,
    ) -> str:
        """
        Build prompt for initial (zero-shot) recommendation.
        
        Args:
            meta_features: Dataset meta-features dictionary
            search_space: Optuna search space or parameter grid
            zt_prior: Optional ZeroTune zero-shot prediction
            
        Returns:
            Formatted prompt string
        """
        similar = retrieve_similar_datasets(
            meta_features, self.kb_path, n_similar=self.n_similar_datasets
        )
        
        model_class = self._model_class_names.get(self.model_type, self.model_type)
        meta_str = self._format_meta_features(meta_features)
        search_str = self._format_search_space(search_space)
        similar_str = format_similar_datasets_for_prompt(similar)
        prior_str = self._format_zt_prior(zt_prior)
        n_samples = meta_features.get('n_samples', 0)
        cond_stats_str = self._format_conditional_stats(n_samples)
        corr_str = self._format_correlations(meta_features)
        
        return f"""You are an expert ML practitioner optimizing hyperparameters for a {model_class}.

{self.param_guide}

{cond_stats_str}

{corr_str}

## Current Dataset Characteristics
{meta_str}

## ZeroTune Zero-Shot Prediction (Baseline Hypothesis)
{prior_str}

## Search Space
{search_str}

## Similar Datasets from Knowledge Base
{similar_str}

## Task
Based on all the above context, recommend the FIRST set of hyperparameters to try.

Consider:
1. The dataset size category and appropriate complexity level (see Historical Patterns)
2. Risk of overfitting given the feature/sample ratio
3. Class imbalance implications
4. Patterns from similar datasets in the knowledge base
5. The ZeroTune baseline hypothesis (does it align with your reasoning?)
6. The parameter interactions and heuristics from the guide

Provide step-by-step reasoning, then your recommended hyperparameters."""

    def build_sequential_prompt(
        self,
        meta_features: dict,
        search_space: dict,
        completed_trials: list,
        zt_prior: Optional[dict] = None,
        kb_confidence: Optional[float] = None,
    ) -> str:
        """Build prompt for sequential recommendation based on trial history."""
        similar = retrieve_similar_datasets(
            meta_features, self.kb_path, n_similar=self.n_similar_datasets
        )
        
        if kb_confidence is None and similar:
            kb_confidence = max(d['similarity'] for d in similar)
        
        best_score = max(t.value for t in completed_trials) if completed_trials else 0.0
        n_trials = len(completed_trials)
        
        model_class = self._model_class_names.get(self.model_type, self.model_type)
        meta_str = self._format_meta_features(meta_features)
        search_str = self._format_search_space(search_space)
        trial_str = self._format_trial_history(completed_trials)
        similar_str = format_similar_datasets_for_prompt(similar)
        prior_str = self._format_zt_prior(zt_prior)
        n_samples = meta_features.get('n_samples', 0)
        cond_stats_str = self._format_conditional_stats(n_samples)
        corr_str = self._format_correlations(meta_features)
        strategy_str = self._format_strategy_recommendation(kb_confidence)
        priority_str = self._format_information_priority(n_trials)
        
        return f"""You are an expert ML practitioner optimizing hyperparameters for a {model_class}.

{self.param_guide}

{cond_stats_str}

{corr_str}

## Current Dataset Characteristics
{meta_str}

## ZeroTune Zero-Shot Prediction (Baseline Hypothesis)
{prior_str}

## Search Space
{search_str}

## Optimization History ({n_trials} trials completed)
Best score so far: {best_score:.4f}

### Trial History (sorted by score, best first)
{trial_str}

## Similar Datasets from Knowledge Base
{similar_str}

{strategy_str}

{priority_str}

## Task
Based on the observed trial results and expert knowledge above, recommend FOUR diverse configurations:

1. **EXPLOIT**: Small refinements to the best configuration (adjust 1-2 params slightly)
2. **EXPLORE_MILD**: A nearby but unexplored region (change 1-2 params more significantly)
3. **EXPLORE_AGGRESSIVE**: A very different region (opposite end of key parameters)
4. **WILDCARD**: High-risk, high-reward (extreme or unusual combinations)

Provide all four recommendations with brief reasoning for each."""

    def _format_information_priority(self, n_trials: int) -> str:
        """Format information priority guidance based on trial count."""
        lines = ["## Information Priority"]
        lines.append("**IMPORTANT**: Weight evidence in this order:")
        lines.append("1. **Current run trials (highest)** - Direct observations from THIS dataset. Trust these most.")
        lines.append("2. **ZeroTune prior & conditional stats** - Model-based predictions trained on the entire KB.")
        lines.append("3. **Similar datasets from KB** - Nearest-neighbor lookup, useful but may not generalize.")
        
        if n_trials >= 5:
            lines.append("")
            lines.append(f"With {n_trials} trials completed, prioritize current run observations over all KB-derived guidance.")
        
        return "\n".join(lines)

    def _format_meta_features(self, meta_features: dict) -> str:
        """Format meta-features for prompt."""
        n_samples = meta_features.get('n_samples', 0)
        n_features = meta_features.get('n_features', 0)
        
        feature_sample_ratio = n_features / n_samples if n_samples > 0 else 0
        
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
                lines.append(f"- {param}: [{spec.low}, {spec.high}]")
            else:
                lines.append(f"- {param}: {spec}")
        
        return "\n".join(lines)
    
    def _format_trial_history(self, trials: list, max_trials: int = 15) -> str:
        """Format Optuna trials for prompt."""
        if not trials:
            return "No trials completed yet."
        
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

    def _format_zt_prior(self, zt_prior: Optional[dict]) -> str:
        """Format ZeroTune prior prediction."""
        if not zt_prior:
            return "No ZeroTune prediction available."
        
        params_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(zt_prior.items())
        )
        return f"Predicted config: {params_str}\n(Use as starting hypothesis, deviate if trial evidence suggests otherwise)"

    def _format_conditional_stats(self, n_samples: int) -> str:
        """Format conditional statistics based on dataset size."""
        size_cat = get_dataset_size_category(n_samples)
        stats = self.conditional_stats.get(size_cat, {})
        
        if not stats:
            return f"No conditional statistics available for {size_cat} datasets."
            
        lines = [f"### Historical Patterns for {size_cat.upper()} Datasets (like yours)"]
        
        for param, stat in sorted(stats.items()):
            lines.append(f"- {param}: median={stat['median']:.3f}, range=[{stat['min']:.3f}, {stat['max']:.3f}]")
            
        return "\n".join(lines)

    def _format_correlations(self, meta_features: dict) -> str:
        """Format parameter correlations relevant to this dataset."""
        if not self.correlations:
            return ""
            
        relevant_corrs = []
        for (mf, param), corr in self.correlations.items():
            if abs(corr) > 0.3:
                relevant_corrs.append((mf, param, corr))
        
        if not relevant_corrs:
            return ""
            
        relevant_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        lines = ["### Relevant Parameter Correlations (Global Patterns)"]
        for mf, param, corr in relevant_corrs[:5]:
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) > 0.5 else "moderate"
            lines.append(f"- {mf} has {strength} {direction} correlation with {param} (r={corr:.2f})")
                
        return "\n".join(lines)

    def _format_strategy_recommendation(self, kb_confidence: Optional[float]) -> str:
        """Format strategy recommendation based on KB confidence."""
        if kb_confidence is None:
            return ""
        
        lines = [f"## Recommended Strategy (KB Confidence: {kb_confidence:.2f})"]
        
        if kb_confidence >= 0.85:
            lines.append("**HIGH CONFIDENCE**: Your dataset is very similar to those in the knowledge base.")
            lines.append("- Prioritize **EXPLOIT**: The KB patterns are likely reliable for your dataset.")
            lines.append("- Use EXPLORE configurations as secondary options to refine near-optimal regions.")
        elif kb_confidence >= 0.60:
            lines.append("**MODERATE CONFIDENCE**: Your dataset has some similarity to the knowledge base.")
            lines.append("- Balance **EXPLOIT** and **EXPLORE_MILD**: Refine based on trials, but stay open to adjustments.")
            lines.append("- KB patterns provide useful guidance but may need adaptation.")
        else:
            lines.append("**LOW CONFIDENCE**: Your dataset differs significantly from the knowledge base.")
            lines.append("- Prioritize **EXPLORE**: Cast a wider net since historical patterns may not apply.")
            lines.append("- Give more weight to **WILDCARD** configurations to discover novel good regions.")
            lines.append("- Rely more on trial observations than KB priors.")
        
        return "\n".join(lines)
