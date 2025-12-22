"""
Static expert knowledge guides for hyperparameter optimization.

These guides provide the LLM with domain expertise about each model type's
hyperparameters, their effects, interactions, and dataset-conditional heuristics.
"""

DECISION_TREE_PARAM_GUIDE = """
## DecisionTreeClassifier (CART) — hyperparameter rules of thumb

IMPORTANT: Only recommend hyperparameters that exist in the provided Search Space.

### Why trees overfit
- Default tree settings often yield fully grown, unpruned trees; explicitly control complexity to avoid overfitting and huge trees.

### Highest-impact knobs (tune in this order)

1) min_samples_leaf (best "anti-overfit" control)
- Start rule-of-thumb: min_samples_leaf = 5 (a common safe initial value).
- If the sample size varies greatly across datasets, prefer a FLOAT percentage (e.g., 0.005 to 0.02) so it scales with n_samples.
- If Feature/Sample ratio is high or target correlations are weak/noisy, increase min_samples_leaf.

2) max_depth (caps complexity)
- If small dataset (< 1000 samples) or feature/sample ratio is high: prefer shallow depth.
- If underfitting: increase max_depth gradually.
- If you have both max_depth and min_samples_leaf available, favor adjusting min_samples_leaf first.

3) ccp_alpha (cost-complexity post-pruning)
- If available, include small positive values (near 0) when you suspect overfitting.
- Larger ccp_alpha prunes more aggressively.

4) min_samples_split (secondary; keep consistent with leaf size)
- Interaction rule: enforce min_samples_split >= 2 * min_samples_leaf when both are in the search space.

### Imbalance heuristic (if class_weight exists in the Search Space)
- If Class imbalance ratio is meaningfully > 1, try class_weight="balanced" early.

### Meta-feature conditioned defaults (for FIRST trial)
- small (< 1000): smaller max_depth, larger min_samples_leaf (or small >0 ccp_alpha)
- medium (1000–10000): moderate depth, min_samples_leaf around 5–20 depending on noise
- large (>=10000): can allow deeper trees, but still avoid tiny leaves if signal is noisy

### What to do based on observed trials
- Overfitting signs (best score occurs with simpler trees): increase min_samples_leaf and/or ccp_alpha; optionally reduce max_depth.
- Underfitting signs (scores plateau low, deeper helps): increase max_depth and/or reduce min_samples_leaf.
"""

RANDOM_FOREST_PARAM_GUIDE = """
## RandomForestClassifier — hyperparameter rules of thumb

IMPORTANT: Only recommend hyperparameters that exist in the provided Search Space.

### Core idea
Random forests average many decision trees built on bootstrap samples and feature subsampling to improve accuracy and control overfitting.

### Highest-impact knobs (tune in this order)

1) max_features (feature subsampling; often the most important RF knob)
- Classic default for classification: sqrt(p) features per split.
- Smaller max_features => more tree diversity (less correlation), can help generalization if many noisy/redundant features.
- Larger max_features => stronger individual trees, can help if signal is weak and features are few.

2) min_samples_leaf / min_samples_split / max_depth (tree regularization inside the forest)
- If overfitting/noisy data: increase min_samples_leaf, optionally cap max_depth.
- If underfitting: allow deeper trees (max_depth=None) and/or reduce min_samples_leaf.

3) n_estimators (number of trees)
- More trees mainly reduces variance; increasing n_estimators is usually "safe" for generalization.
- Avoid very small n_estimators; use enough trees for stability (compute permitting).

4) max_samples (only if bootstrap=True and max_samples is in Search Space)
- Default draws n_samples for each tree when max_samples=None.
- Smaller max_samples can increase diversity and speed up training, sometimes improving generalization.

### Imbalance heuristic (if class_weight exists in the Search Space)
- If Class imbalance ratio is meaningfully > 1, try:
  - class_weight="balanced_subsample" early (weights computed per bootstrap sample),
  - or class_weight="balanced".

### OOB (if oob_score exists in Search Space)
- If bootstrap=True, enabling oob_score can provide a cheap generalization estimate during HPO.

### Meta-feature conditioned defaults (for FIRST trial)
- small (< 1000): keep trees simpler (larger min_samples_leaf and/or capped max_depth) and use max_features="sqrt".
- medium (1000–10000): max_features="sqrt" is a strong baseline; tune leaf size next.
- large (>=10000): allow deeper trees but still tune max_features and leaf size; increase n_estimators if variance is high.

### What to do based on observed trials
- If best trials use smaller max_features and larger leaf sizes: keep exploiting that region (reduce correlation + smooth trees).
- If best trials use larger max_features and deeper trees: exploit complexity (but watch for diminishing returns).
"""

XGBOOST_PARAM_GUIDE = """
## XGBoostClassifier (booster=gbtree/dart) — hyperparameter rules of thumb

IMPORTANT: Only recommend hyperparameters that exist in the provided Search Space.

### Mental model
XGBoost adds trees sequentially; it can overfit if trees are too complex or if boosting runs too long.
Key levers: tree complexity (max_depth, min_child_weight, gamma), randomness (subsample/colsample_*), regularization (reg_alpha/reg_lambda), and shrinkage (learning_rate).

### Highest-impact knobs (tune in this order)

1) learning_rate (eta) + n_estimators (boosting length)
- learning_rate (alias: eta) is "step size shrinkage" that makes boosting more conservative. Lower learning_rate typically needs more boosting rounds. (eta range [0,1]).
- If you can, rely on early stopping with a validation set to choose the effective number of trees/rounds.

2) max_depth (tree depth) + min_child_weight (minimum child hessian)
- max_depth increases model complexity and overfitting risk; 0 means "no limit" (also can increase memory use a lot).
- min_child_weight: larger values are more conservative and stop splitting when children don't have enough "weight" (hessian).
- Heuristic: If Feature/Sample ratio is high or dataset is small, prefer smaller max_depth and larger min_child_weight.

3) subsample + colsample_bytree (and colsample_bylevel/bynode if exposed)
- subsample: fraction of rows used per boosting iteration; 0.5 means sample half the rows and "will prevent overfitting"; range (0,1]. Typically set subsample >= 0.5 for good results.
- colsample_*: fraction of columns subsampled; range (0,1], and these parameters work cumulatively. Lower values increase randomness and can reduce overfitting.

4) gamma (min_split_loss)
- Minimum loss reduction required to split; larger gamma is more conservative.

5) reg_lambda (L2) and reg_alpha (L1)
- Increasing reg_lambda (alias: lambda) and/or reg_alpha (alias: alpha) makes the model more conservative.

### Imbalance heuristics
- scale_pos_weight controls positive/negative class balance; typical value: sum(negative) / sum(positive).
- max_delta_step can help in logistic regression when classes are extremely imbalanced; setting it to 1–10 may help make updates more conservative.

### Other useful knobs (when in Search Space)
- booster: can be gbtree, gblinear, dart. (Most common is gbtree; dart adds dropout-style boosting which can help regularization.)
- tree_method: auto is the same as hist in current docs; hist is a fast histogram-based method.

### Meta-feature conditioned defaults (for FIRST trial)
- small (< 1000 samples) or high Feature/Sample ratio:
  - conservative trees: smaller max_depth, larger min_child_weight
  - add randomness: subsample ~0.7–0.9, colsample_bytree ~0.7–0.9 (if available)
  - consider some regularization (reg_lambda > 1 or small reg_alpha) if search space allows
- medium (1000–10000):
  - moderate depth; tune (max_depth, min_child_weight) first, then subsample/colsample
- large (>=10000):
  - can allow deeper trees, but still tune subsample/colsample to control variance and speed

### What to do based on observed trials
- Overfitting signs (simpler settings score better):
  - decrease max_depth, increase min_child_weight and/or gamma
  - reduce subsample/colsample_* (more randomness)
  - increase reg_lambda/reg_alpha
  - increase learning_rate slightly only if under-training, otherwise prefer early stopping / fewer estimators
- Underfitting signs (scores low and improve with complexity):
  - increase max_depth and/or reduce min_child_weight/gamma
  - increase colsample/subsample toward 1.0
  - decrease reg_* if overly constrained
"""


def get_param_guide(model_type: str) -> str:
    """Get the parameter guide for a specific model type."""
    guides = {
        'decision_tree': DECISION_TREE_PARAM_GUIDE,
        'random_forest': RANDOM_FOREST_PARAM_GUIDE,
        'xgboost': XGBOOST_PARAM_GUIDE,
    }
    return guides.get(model_type.lower(), "")
