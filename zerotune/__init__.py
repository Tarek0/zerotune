"""
ZeroTune: Zero-shot hyperparameter optimization using pre-trained models.

ZeroTune provides instant hyperparameter predictions for machine learning models
using pre-trained models trained on diverse datasets. No optimization time required!

Usage example:
```python
from zerotune import ZeroTunePredictor
from xgboost import XGBClassifier
import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Get optimal hyperparameters instantly using pre-trained model
predictor = ZeroTunePredictor(model_name='xgboost', task_type='binary')
best_params = predictor.predict(X, y)

# Train model with predicted hyperparameters
model = XGBClassifier(**best_params)
model.fit(X, y)
```

For building knowledge bases to train new predictors, use the ZeroTune class.
"""

from zerotune.predictors import ZeroTunePredictor
from zerotune.core import (
    ZeroTune,
    ModelConfigs,
    fetch_open_ml_data,
    prepare_data,
    get_dataset_ids,
    get_recommended_datasets
)

__version__ = '0.1.0'

__all__ = [
    'ZeroTunePredictor',  # Main interface for zero-shot predictions
    'ZeroTune',           # For building knowledge bases
    'ModelConfigs',
    'fetch_open_ml_data',
    'prepare_data',
    'get_dataset_ids',
    'get_recommended_datasets'
] 