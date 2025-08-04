# Pretrained Models for ZeroTune

This directory contains pretrained models that can be used for zero-shot hyperparameter prediction.

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

This model predicts optimal hyperparameters for XGBoost models. It predicts the following hyperparameters:

- `n_estimators`: Number of gradient boosted trees
- `max_depth`: Maximum depth of the trees
- `learning_rate`: Step size shrinkage used to prevent overfitting
- `subsample`: Subsample ratio of the training instances
- `colsample_bytree`: Subsample ratio of columns when constructing each tree
- `gamma`: Minimum loss reduction required to make a further partition

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

The models are trained to optimize the ROC AUC metric for classification tasks and R² score for regression tasks. They typically provide a significant improvement over the default hyperparameters, especially for datasets with:

- High-dimensional feature spaces
- Class imbalance
- Complex decision boundaries

#### Training Data

These models were trained on a diverse set of datasets, including:
- Synthetic datasets with varying characteristics
- Classification and regression datasets from OpenML
- Real-world problems

## Adding Your Own Models

You can add your own pretrained models to this directory. Make sure they follow the expected structure:

```python
model_data = {
    'model': trained_model,  # The RandomForestRegressor instance
    'dataset_features': dataset_features,  # List of feature names
    'target_params': target_params,  # List of target parameter names
    'score': score  # The model's score
}

# Save the model
import joblib
joblib.dump(model_data, "your_model_name.joblib")
```

Then, add the model to the `AVAILABLE_MODELS` dictionary in `zerotune/predictors.py` to make it available via the `ZeroTunePredictor` class. 