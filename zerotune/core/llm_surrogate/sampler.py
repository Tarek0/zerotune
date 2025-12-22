"""
LLM Surrogate Sampler for Optuna.

A custom Optuna sampler that uses an LLM as the surrogate model for
hyperparameter optimization, replacing traditional methods like TPE.
"""

import json
import numpy as np
from typing import Any, Optional, Sequence
import optuna
from optuna.samplers import BaseSampler
from optuna.distributions import (
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from zerotune.core.llm_surrogate.prompt_builder import PromptBuilder
from zerotune.core.llm_surrogate.schemas import HPORecommendation, InitialRecommendation


class LLMSurrogateSampler(BaseSampler):
    """
    Optuna sampler using LLM as surrogate model.
    
    This sampler uses a Large Language Model to reason about the hyperparameter
    search space and trial history to suggest the next configuration to try.
    
    Attributes:
        meta_features: Dataset meta-features for context
        model_type: Type of ML model being optimized
        reasoning_history: List of LLM reasoning outputs for analysis
    """
    
    def __init__(
        self,
        meta_features: dict,
        kb_path: str,
        model_type: str = "decision_tree",
        llm_model: str = "gpt-5-mini-2025-08-07",
        n_similar_datasets: int = 3,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        n_zerotune_initial: int = 0,
        n_random_initial: int = 0,
    ):
        """
        Initialize the LLM Surrogate Sampler.
        
        Args:
            meta_features: Dataset meta-features dictionary
            kb_path: Path to the knowledge base JSON file
            model_type: Type of model ('decision_tree', 'random_forest', 'xgboost')
            llm_model: OpenAI model to use (e.g., 'gpt-5-mini-2025-08-07', 'gpt-4o-mini')
            n_similar_datasets: Number of similar datasets to retrieve for context
            temperature: LLM temperature (default: 1.0 for gpt-5-nano, can use 0.0 for other models)
            seed: Random seed for fallback sampling
            n_zerotune_initial: Number of initial trials using ZeroTune prediction (default: 0).
                Set to 1+ to warm-start with ZeroTune's zero-shot prediction.
                Set to 0 to skip ZeroTune and start with LLM or random.
            n_random_initial: Number of trials using random sampling for exploration (default: 0).
                Provides diverse starting points before LLM sequential optimization.
        """
        self._meta_features = meta_features
        self._kb_path = kb_path
        self._model_type = model_type
        self._llm_model = llm_model
        self._temperature = temperature
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        
        # Hybrid sampling parameters
        self._n_zerotune_initial = n_zerotune_initial
        self._n_random_initial = n_random_initial
        self._zerotune_prediction = None  # Cached ZeroTune prediction
        
        # Initialize prompt builder
        self._prompt_builder = PromptBuilder(
            model_type=model_type,
            kb_path=kb_path,
            n_similar_datasets=n_similar_datasets,
        )
        
        # Initialize LLM (lazy loading)
        self._llm = None
        self._structured_llm_initial = None
        self._structured_llm_sequential = None
        
        # Store reasoning history for analysis
        self.reasoning_history: list[dict] = []
        
        # Fallback/random sampler
        self._fallback_sampler = optuna.samplers.RandomSampler(seed=seed)
        
        # Cache for batch recommendations (4 configs per LLM call)
        self._pending_configs: list[dict] = []
        self._last_llm_call_trial: int = -1
    
    def _init_llm(self):
        """Lazy initialization of LLM to avoid import errors if not used."""
        if self._llm is not None:
            return
        
        try:
            from langchain_openai import ChatOpenAI
            
            self._llm = ChatOpenAI(
                model=self._llm_model,
                temperature=self._temperature,
            )
            
            # Create structured output versions with strict JSON schema enforcement
            self._structured_llm_initial = self._llm.with_structured_output(
                InitialRecommendation,
                method="json_schema",
                strict=True,
            )
            self._structured_llm_sequential = self._llm.with_structured_output(
                HPORecommendation,
                method="json_schema",
                strict=True,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for LLMSurrogateSampler. "
                "Install with: pip install langchain-openai"
            ) from e
    
    def _get_zerotune_prediction(self) -> dict:
        """
        Get ZeroTune's zero-shot prediction for initial trial.
        
        Uses the pre-trained predictor model to get a strong starting point.
        Returns cached prediction if already computed.
        """
        if self._zerotune_prediction is not None:
            return self._zerotune_prediction
        
        try:
            import joblib
            from pathlib import Path
            
            # Map model type to predictor file
            predictor_files = {
                'decision_tree': 'predictor_decision_tree_dt_kb_v1_full.joblib',
                'random_forest': 'predictor_random_forest_rf_kb_v1_full.joblib',
                'xgboost': 'predictor_xgboost_xgb_kb_v1_full.joblib',
            }
            
            predictor_filename = predictor_files.get(self._model_type)
            if not predictor_filename:
                print(f"No ZeroTune predictor for {self._model_type}, using random")
                return None
            
            # Find predictor file (models dir is at project root, not inside zerotune package)
            package_dir = Path(__file__).parent.parent.parent  # zerotune package
            project_root = package_dir.parent  # project root
            models_dir = project_root / "models"
            predictor_path = models_dir / predictor_filename
            
            if not predictor_path.exists():
                print(f"ZeroTune predictor not found: {predictor_path}")
                return None
            
            # Load predictor model
            model_data = joblib.load(predictor_path)
            
            # Extract features (handle both key naming conventions)
            feature_names = model_data.get('feature_names') or model_data.get('dataset_features', [])
            feature_vector = []
            
            for feature_name in feature_names:
                value = self._meta_features.get(feature_name, 0.0)
                feature_vector.append(float(value) if value is not None else 0.0)
            
            feature_vector = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
            
            # Normalize features
            if 'normalization_params' in model_data:
                norm_params = model_data['normalization_params']
                for i, feature_name in enumerate(feature_names):
                    if feature_name in norm_params:
                        norm_info = norm_params[feature_name]
                        min_val = norm_info.get('min', 0.0)
                        range_val = norm_info.get('range', 1.0)
                        if range_val > 0:
                            normalized = (feature_vector[0, i] - min_val) / range_val
                            feature_vector[0, i] = np.clip(normalized, -1000.0, 1000.0)
                        else:
                            feature_vector[0, i] = 0.0
            
            # Apply feature selection
            if 'feature_selector' in model_data and model_data['feature_selector'] is not None:
                feature_vector = model_data['feature_selector'].transform(feature_vector)
            
            # Predict
            prediction = model_data['model'].predict(feature_vector)[0]
            
            # Convert to hyperparameters (handle both key naming conventions)
            target_params = model_data.get('param_names') or model_data.get('target_params', [])
            hyperparams = {}
            
            for i, param_name in enumerate(target_params):
                clean_name = param_name.replace('params_', '')
                if i < len(prediction):
                    value = prediction[i]
                    # Type conversion based on parameter
                    if clean_name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                        value = max(1, int(round(value)))
                    elif clean_name in ['learning_rate', 'subsample', 'colsample_bytree', 'gamma']:
                        value = max(0.01, min(1.0, float(value)))
                    hyperparams[clean_name] = value
            
            # Handle criterion for decision tree (categorical)
            if self._model_type == 'decision_tree' and 'criterion' not in hyperparams:
                hyperparams['criterion'] = 'gini'
            
            self._zerotune_prediction = hyperparams
            
            # Log the prediction
            self.reasoning_history.append({
                'trial': 0,
                'type': 'zerotune',
                'strategy': 'zerotune-warmstart',
                'reasoning': 'Using ZeroTune zero-shot prediction as initial warm-start',
                'params': hyperparams,
                'confidence': 0.9,
            })
            
            return hyperparams
            
        except Exception as e:
            print(f"ZeroTune prediction failed: {e}")
            return None
    
    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        """Infer the search space from the study."""
        # Get completed trials from the study
        completed_trials = [
            t for t in study.trials
            if t.state == TrialState.COMPLETE
        ]
        
        if not completed_trials:
            # No completed trials yet, return empty (will use sample_independent)
            return {}
        
        return optuna.search_space.intersection_search_space(completed_trials)
    
    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """
        Sample parameters using hybrid 3-phase approach.
        
        Phase 1 (trials 0 to n_zerotune_initial-1): ZeroTune predictions
        Phase 2 (trials n_zerotune_initial to n_zerotune_initial+n_random_initial-1): Random exploration
        Phase 3 (remaining trials): LLM reasoning
        
        Args:
            study: Optuna study object
            trial: Current trial
            search_space: Dictionary of parameter distributions
            
        Returns:
            Dictionary of sampled parameter values
        """
        # Get completed trials
        n_completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        
        # Determine which phase we're in
        phase1_end = self._n_zerotune_initial
        phase2_end = phase1_end + self._n_random_initial
        
        # Phase 1: ZeroTune warm-start
        if n_completed < phase1_end:
            zt_params = self._get_zerotune_prediction()
            if zt_params is not None:
                validated = self._validate_params(zt_params, search_space)
                return validated
            # Fall through to Phase 2 or 3 if ZeroTune fails
        
        # Phase 2: Random exploration
        if n_completed < phase2_end:
            self.reasoning_history.append({
                'trial': n_completed,
                'type': 'random',
                'strategy': 'exploration',
                'reasoning': f'Random exploration trial {n_completed - phase1_end + 1}/{self._n_random_initial}',
                'confidence': 0.5,
            })
            return self._fallback_sampler.sample_relative(study, trial, search_space)
        
        # Phase 3: LLM reasoning
        self._init_llm()
        
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        simple_search_space = self._convert_search_space(search_space)
        
        try:
            if len(completed_trials) == 0:
                # No history yet - use LLM initial recommendation
                # Pass ZeroTune prediction as prior if available
                zt_params = self._get_zerotune_prediction()
                params = self._recommend_initial(simple_search_space, zt_prior=zt_params)
            else:
                # Sequential: LLM reasons based on full history
                # Pass ZeroTune prediction as prior if available
                zt_params = self._get_zerotune_prediction()
                params = self._recommend_next(simple_search_space, completed_trials, zt_prior=zt_params)
            
            validated_params = self._validate_params(params, search_space)
            return validated_params
            
        except Exception as e:
            print(f"LLM sampling failed: {e}. Falling back to random sampling.")
            self.reasoning_history.append({
                'trial': trial.number,
                'type': 'fallback',
                'error': str(e),
                'fallback': True,
            })
            return self._fallback_sampler.sample_relative(study, trial, search_space)
    
    def _recommend_initial(self, search_space: dict, zt_prior: Optional[dict] = None) -> dict:
        """Get initial (zero-shot) recommendation from LLM."""
        prompt = self._prompt_builder.build_initial_prompt(
            meta_features=self._meta_features,
            search_space=search_space,
            zt_prior=zt_prior,
        )
        
        # Call LLM with structured output
        response: InitialRecommendation = self._structured_llm_initial.invoke(prompt)
        
        # Convert hyperparameters to dict
        params_dict = response.hyperparameters.model_dump()
        
        # Store reasoning
        self.reasoning_history.append({
            'trial': 0,
            'type': 'initial',
            'dataset_assessment': response.dataset_assessment,
            'strategy': response.strategy,
            'reasoning': response.reasoning_summary,
            'params': params_dict,
            'confidence': response.confidence,
        })
        
        return params_dict
    
    def _recommend_next(self, search_space: dict, completed_trials: list, zt_prior: Optional[dict] = None) -> dict:
        """Get sequential recommendation from LLM based on trial history.
        
        Uses batch strategy (4 configs per LLM call):
        - On LLM call: Returns exploit, caches explore_mild, explore_aggressive, wildcard
        - On subsequent 3 trials: Returns cached configs without LLM call
        """
        n_completed = len(completed_trials)
        
        # Check if we have cached configs to use
        if self._pending_configs:
            cached = self._pending_configs.pop(0)
            params_dict = cached['params']
            
            self.reasoning_history.append({
                'trial': n_completed,
                'type': cached['type'],
                'strategy': f"{cached['type']} (cached)",
                'reasoning': cached['reasoning'],
                'params': params_dict,
                'confidence': cached['confidence'],
            })
            return params_dict
        
        # Call LLM to get all 4 recommendations
        prompt = self._prompt_builder.build_sequential_prompt(
            meta_features=self._meta_features,
            search_space=search_space,
            completed_trials=completed_trials,
            zt_prior=zt_prior,
        )
        
        response: HPORecommendation = self._structured_llm_sequential.invoke(prompt)
        
        # Extract all 4 recommendations
        exploit_params = response.exploit.hyperparameters.model_dump()
        explore_mild_params = response.explore_mild.hyperparameters.model_dump()
        explore_aggressive_params = response.explore_aggressive.hyperparameters.model_dump()
        wildcard_params = response.wildcard.hyperparameters.model_dump()
        
        # Cache the next 3 configs (explore_mild, explore_aggressive, wildcard)
        self._pending_configs = [
            {'type': 'explore_mild', 'params': explore_mild_params, 
             'reasoning': response.explore_mild.reasoning, 'confidence': response.explore_mild.confidence},
            {'type': 'explore_aggressive', 'params': explore_aggressive_params,
             'reasoning': response.explore_aggressive.reasoning, 'confidence': response.explore_aggressive.confidence},
            {'type': 'wildcard', 'params': wildcard_params,
             'reasoning': response.wildcard.reasoning, 'confidence': response.wildcard.confidence},
        ]
        self._last_llm_call_trial = n_completed
        
        # Store reasoning for exploit (used now)
        self.reasoning_history.append({
            'trial': n_completed,
            'type': 'exploit',
            'overall_assessment': response.overall_assessment,
            'trial_analyses': [
                {'trial': ta.trial_number, 'observation': ta.observation, 'implication': ta.implication}
                for ta in response.trial_analyses
            ],
            'strategy': 'exploit',
            'reasoning': response.exploit.reasoning,
            'params': exploit_params,
            'confidence': response.exploit.confidence,
            'batch_cached': ['explore_mild', 'explore_aggressive', 'wildcard'],
        })
        
        return exploit_params
    
    def _convert_search_space(self, search_space: dict[str, BaseDistribution]) -> dict:
        """Convert Optuna distributions to simple format for prompt."""
        simple_space = {}
        
        for name, dist in search_space.items():
            if isinstance(dist, FloatDistribution):
                if dist.log:
                    simple_space[name] = (dist.low, dist.high, 'log')
                else:
                    simple_space[name] = (dist.low, dist.high, 'linear')
            elif isinstance(dist, IntDistribution):
                simple_space[name] = (dist.low, dist.high, 'int')
            elif isinstance(dist, CategoricalDistribution):
                simple_space[name] = list(dist.choices)
            else:
                simple_space[name] = str(dist)
        
        return simple_space
    
    def _validate_params(
        self,
        params: dict,
        search_space: dict[str, BaseDistribution]
    ) -> dict:
        """Validate and clip parameters to search space bounds."""
        validated = {}
        
        for name, dist in search_space.items():
            if name not in params:
                # Parameter missing - sample randomly
                validated[name] = self._sample_random(dist)
                continue
            
            value = params[name]
            
            if isinstance(dist, FloatDistribution):
                # Clip to bounds
                value = float(value)
                value = max(dist.low, min(dist.high, value))
                validated[name] = value
                
            elif isinstance(dist, IntDistribution):
                # Clip to bounds and convert to int
                value = int(round(float(value)))
                value = max(dist.low, min(dist.high, value))
                validated[name] = value
                
            elif isinstance(dist, CategoricalDistribution):
                # Check if value is valid choice
                if value in dist.choices:
                    validated[name] = value
                else:
                    # Try to find closest match or use random
                    validated[name] = self._rng.choice(dist.choices)
            else:
                validated[name] = value
        
        return validated
    
    def _sample_random(self, dist: BaseDistribution) -> Any:
        """Sample a random value from distribution."""
        if isinstance(dist, FloatDistribution):
            if dist.log:
                log_low = np.log(dist.low)
                log_high = np.log(dist.high)
                return float(np.exp(self._rng.uniform(log_low, log_high)))
            else:
                return float(self._rng.uniform(dist.low, dist.high))
        elif isinstance(dist, IntDistribution):
            return int(self._rng.randint(dist.low, dist.high + 1))
        elif isinstance(dist, CategoricalDistribution):
            return self._rng.choice(dist.choices)
        else:
            raise ValueError(f"Unknown distribution type: {type(dist)}")
    
    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a single parameter independently (fallback)."""
        return self._sample_random(param_distribution)
    
    def get_reasoning_summary(self) -> str:
        """Get a summary of all reasoning from the optimization run."""
        if not self.reasoning_history:
            return "No trials completed yet."
        
        lines = ["=== LLM Reasoning Summary ===\n"]
        
        for entry in self.reasoning_history:
            if entry.get('error'):
                lines.append(f"Trial {entry['trial']}: ERROR - {entry['error']}")
            else:
                lines.append(f"Trial {entry['trial']} ({entry['type']}):")
                lines.append(f"  Strategy: {entry['strategy']}")
                lines.append(f"  Reasoning: {entry['reasoning']}")
                lines.append(f"  Confidence: {entry['confidence']:.2f}")
                lines.append(f"  Params: {entry['params']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def save_reasoning_history(self, path: str):
        """Save reasoning history to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.reasoning_history, f, indent=2)
