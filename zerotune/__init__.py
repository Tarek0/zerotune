"""
ZeroTune: Automatic hyperparameter optimization using meta-learning.

ZeroTune is a Python package that optimizes hyperparameters for machine
learning models by leveraging knowledge from similar datasets, effectively
reducing the search space and optimization time.

Usage example:
```python
from zerotune import ZeroTune
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize ZeroTune with model type
zt = ZeroTune(model_type='decision_tree')

# Optimize hyperparameters and get the best model
best_params, best_score, model = zt.optimize(X, y)
```
"""

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
    'ZeroTune',
    'ModelConfigs',
    'fetch_open_ml_data',
    'prepare_data',
    'get_dataset_ids',
    'get_recommended_datasets'
] 