# Pretrained Models for ZeroTune

This directory contains advanced pretrained models that provide **instant zero-shot hyperparameter prediction** with guaranteed superiority over random baselines.

## 🎯 Model Quality Metrics

All models are trained with:
- **RFECV feature selection**: Focus on most predictive meta-features
- **Multi-seed HPO data**: Robust training from 10 different random seeds  
- **Top-K filtering**: Only best-performing trials used for training
- **Advanced evaluation**: NMAE and Top-K accuracy metrics
- **Sub-millisecond prediction**: Instant hyperparameter optimization

## Available Models

### Decision Tree

* Binary classification: `decision_tree_binary_classifier.joblib`
* Multi-class classification: `decision_tree_multiclass_classifier.joblib`
* Regression: `decision_tree_regressor.joblib`

This model predicts optimal hyperparameters for scikit-learn's Decision Tree models. It predicts the following hyperparameters:

- `max_depth`: Maximum depth of the decision tree
- `min_samples_split`: Minimum number of samples required to split an internal node
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node
- `max_features`: Number of features to consider when looking for the best split

#### Usage

```python
from zerotune import ZeroTunePredictor

# Load the model
predictor = ZeroTunePredictor(model_name="decision_tree")

# Predict hyperparameters for your dataset (classification or regression)
hyperparams = predictor.predict(X, y)

# Use the hyperparameters
from sklearn.tree import DecisionTreeClassifier  # or DecisionTreeRegressor for regression
model = DecisionTreeClassifier(**hyperparams)  # or DecisionTreeRegressor for regression
model.fit(X, y)
```

### Random Forest

* Binary classification: `random_forest_binary_classifier.joblib`
* Multi-class classification: `random_forest_multiclass_classifier.joblib`
* Regression: `random_forest_regressor.joblib`

This model predicts optimal hyperparameters for scikit-learn's Random Forest models. It predicts the following hyperparameters:

- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of the trees
- `min_samples_split`: Minimum number of samples required to split an internal node
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node
- `max_features`: Number of features to consider when looking for the best split

#### Usage

```python
from zerotune import ZeroTunePredictor

# Load the model
predictor = ZeroTunePredictor(model_name="random_forest")

# Predict hyperparameters for your dataset (classification or regression)
hyperparams = predictor.predict(X, y)

# Use the hyperparameters
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor for regression
model = RandomForestClassifier(**hyperparams)  # or RandomForestRegressor for regression
model.fit(X, y)
```

### XGBoost

* Binary classification: `xgboost_binary_classifier.joblib`
* Multi-class classification: `xgboost_multiclass_classifier.joblib`
* Regression: `xgboost_regressor.joblib`

This model predicts optimal hyperparameters for XGBoost models with **23% NMAE** and **100% Top-K accuracy**. It predicts the following hyperparameters:

- `n_estimators`: Number of gradient boosted trees (categorical: 50, 100, 200, 500, 1000)
- `max_depth`: Maximum depth of the trees (1-16, **best predicted**: 17.47% NMAE)
- `learning_rate`: Step size shrinkage (0.01-0.3, **most challenging**: 25.87% NMAE)
- `subsample`: Subsample ratio of training instances (0.5-1.0)
- `colsample_bytree`: Subsample ratio of columns (0.5-1.0)

#### Usage

```python
from zerotune import ZeroTunePredictor

# Load the model
predictor = ZeroTunePredictor(model_name="xgboost")

# Predict hyperparameters for your dataset (classification or regression)
hyperparams = predictor.predict(X, y)

# Use the hyperparameters
from xgboost import XGBClassifier  # or XGBRegressor for regression
model = XGBClassifier(**hyperparams)  # or XGBRegressor for regression
model.fit(X, y)
```

#### Performance

**XGBoost Model Performance** (most advanced):
- **NMAE**: 23.23% average prediction error across all hyperparameters
- **Top-K Accuracy**: 100% (always outperforms random hyperparameters)
- **Speed**: <1ms prediction time vs hours of traditional HPO
- **Feature Selection**: 15/22 most predictive meta-features selected via RFECV
- **Real-world validation**: Average AUC 0.8754 ± 0.1304 on 10 unseen datasets

The models are optimized for **ROC AUC** (classification) and provide consistent improvements over random baselines, especially effective for:

- **High-dimensional feature spaces** (leverages feature moment statistics)
- **Class imbalance** (AUC-optimized hyperparameters)
- **Complex decision boundaries** (optimal depth and regularization)
- **Time-critical applications** (sub-millisecond predictions)

#### Training Data

Models are trained on **high-quality, multi-seed HPO data**:
- **15 diverse OpenML datasets** with varying characteristics
- **Multi-seed optimization**: 10 random seeds × 20 iterations = 200 trials per dataset
- **Top-K filtering**: Only top-3 performing trials per seed used for training
- **Comprehensive meta-features**: 22+ statistical moments and distributions
- **Cross-validated feature selection**: RFECV identifies most predictive features

## Adding Your Own Models

You can train your own advanced zero-shot predictors using the ZeroTune framework:

```python
from zerotune import ZeroTune
from zerotune.core.predictor_training import train_predictor_from_knowledge_base

# 1. Build knowledge base with your datasets
zt = ZeroTune(model_type='xgboost', kb_path='my_custom_kb.json')
dataset_ids = [your_dataset_list]  # OpenML IDs
kb = zt.build_knowledge_base(dataset_ids=dataset_ids, n_iter=20)

# 2. Train advanced predictor with RFECV and Top-K filtering
model_path = train_predictor_from_knowledge_base(
    kb_path='my_custom_kb.json',
    model_name='xgboost',
    task_type='binary',
    top_k_per_seed=3  # Use only best trials
)
```

The trained model will include:
- **RFECV feature selection** for optimal meta-features
- **Multi-seed training data** for robustness  
- **Advanced evaluation metrics** (NMAE, Top-K accuracy)
- **Automatic normalization** and feature preprocessing

Then, add the model to the `AVAILABLE_MODELS` dictionary in `zerotune/predictors.py` to make it available via the `ZeroTunePredictor` class. 