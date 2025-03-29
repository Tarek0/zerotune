# ZeroTune

ZeroTune is a hyperparameter optimization system using meta-learning. It collects dataset meta-features and builds a knowledge base of previous optimizations to predict optimal hyperparameters for new datasets, significantly reducing the search space and optimization time.

## Features

- Extract comprehensive dataset meta-features
- Build model-specific knowledge bases from optimization results
- Find similar datasets using nearest neighbor search on meta-features
- Support for multiple ML models (currently `decision_tree`, `random_forest`, `xgboost`)
- User-friendly API with the `ZeroTune` class as the main interface
- Modular design for easy extension and maintenance

## Installation

This project uses Poetry for dependency management. To install:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository and navigate to it
git clone https://github.com/yourusername/zerotune.git
cd zerotune

# Install dependencies
poetry install
```

## Running ZeroTune CLI Commands

After installation, you can run ZeroTune CLI commands in several ways:

### For Poetry 2.0+ (Recommended)

Poetry 2.0+ no longer includes the `shell` command by default. Use these approaches instead:

```bash
# Option 1: Use poetry run (simplest approach)
poetry run zerotune demo --model xgboost
poetry run zerotune predict --dataset 40981 --model decision_tree
poetry run zerotune datasets --list

# Option 2: Activate the Poetry virtual environment manually
# First, get your virtual environment path
poetry env info

# Then activate it using source
source /path/to/your/virtualenv/bin/activate  # The path from poetry env info

# Now you can run zerotune commands directly
zerotune demo --model xgboost
zerotune predict --dataset 40981 --model decision_tree
```

### For Poetry 1.x

```bash
# Option 1: Use poetry run
poetry run zerotune demo --model xgboost

# Option 2: Activate Poetry shell first
poetry shell
zerotune demo --model xgboost
```

### Using pip (Alternative)

If you prefer pip, you can install ZeroTune using:

```bash
# Install in development mode
pip install -e .

# Run ZeroTune
zerotune demo --model xgboost
```

### Python Module Execution

Run ZeroTune as a Python module without installation:

```bash
python -m zerotune demo --model xgboost
python -m zerotune predict --dataset 40981 --model decision_tree
```

## API Usage

### Basic Usage

```python
from zerotune import ZeroTune
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize ZeroTune with the desired model type
zt = ZeroTune(model_type='decision_tree')

# Optimize hyperparameters
best_params, best_score, model = zt.optimize(X_train, y_train)

# Evaluate on test set
test_score = model.score(X_test, y_test)
print(f"Best parameters: {best_params}")
print(f"Test score: {test_score}")
```

### Using Configuration and Utilities

ZeroTune provides a centralized configuration module and utilities to help manage settings and perform common operations:

```python
from zerotune import ZeroTune
from zerotune.core import CONFIG
from zerotune.core.utils import safe_json_serialize, save_json

# Use configuration values
print(f"Default model type: {CONFIG['defaults']['model_type']}")
print(f"Supported models: {CONFIG['supported']['models']}")

# Initialize ZeroTune with default model type 
zt = ZeroTune()  # Will use the default model_type from CONFIG

# Save results with proper JSON serialization
results = {
    "parameters": best_params,
    "score": best_score,
    "numpy_array": X_train.values[0]  # Will be properly serialized
}
save_json(results, "results.json")
```

### Building a Knowledge Base

```python
from zerotune import ZeroTune, get_recommended_datasets

# Get recommended datasets for building a knowledge base
dataset_ids = get_recommended_datasets(n_datasets=5)

# Initialize ZeroTune with desired model type
zt = ZeroTune(model_type='random_forest')

# Build knowledge base
kb = zt.build_knowledge_base(dataset_ids=dataset_ids, n_iter=20)

# Now you can use the knowledge base for future optimizations
```

### Working with OpenML Datasets

```python
from zerotune import ZeroTune, fetch_open_ml_data, prepare_data

# Fetch a dataset from OpenML
df, target_name, dataset_name = fetch_open_ml_data(dataset_id=40981)
X, y = prepare_data(df, target_name)

# Initialize ZeroTune
zt = ZeroTune(model_type='xgboost')

# Optimize hyperparameters
best_params, best_score, model = zt.optimize(X, y)
```

## CLI Commands Reference

ZeroTune provides several command-line interfaces:

### Demo Command

Run a demonstration of ZeroTune with a sample dataset:

```bash
zerotune demo --model xgboost
```

### Predict Command

Predict optimal hyperparameters for a dataset:

```bash
zerotune predict --dataset 40981 --model decision_tree
zerotune predict --dataset path/to/your/dataset.csv --model random_forest
```

### Train Command

Build a knowledge base from multiple datasets:

```bash
zerotune train --datasets 31 38 44 --model random_forest
```

### Datasets Command

List available OpenML datasets or get recommendations:

```bash
# Show recommended datasets
zerotune datasets

# List all available datasets
zerotune datasets --list

# Filter by category
zerotune datasets --list --category binary
```

## Dataset Catalog

ZeroTune includes a dataset catalog that helps you select appropriate OpenML datasets for training and testing:

```python
from zerotune import get_recommended_datasets, get_dataset_ids
from zerotune.core.data_loading import load_dataset_catalog

# Load the dataset catalog
catalog = load_dataset_catalog()

# Get dataset IDs by category
binary_datasets = get_dataset_ids(category='binary')
multiclass_datasets = get_dataset_ids(category='multi-class')

# Get recommended datasets for training
recommended_datasets = get_recommended_datasets(n_datasets=5)
```

## Module Structure

ZeroTune has a modular structure for easy maintenance and extension:

- `zerotune/core/zero_tune.py` - Main `ZeroTune` class
- `zerotune/core/model_configs.py` - Model configurations
- `zerotune/core/data_loading.py` - Dataset loading functions
- `zerotune/core/feature_extraction.py` - Meta-feature extraction
- `zerotune/core/optimization.py` - Hyperparameter optimization
- `zerotune/core/knowledge_base.py` - Knowledge base management
- `zerotune/core/config.py` - Centralized configuration settings
- `zerotune/core/utils.py` - Common utility functions

## How It Works

1. **Dataset Feature Extraction**: ZeroTune extracts meta-features from datasets, including size, imbalance ratio, feature correlations, and statistical moments.

2. **Knowledge Base Building**: The system builds a knowledge base by optimizing hyperparameters on a collection of datasets and recording the results.

3. **Similar Dataset Finding**: For new datasets, ZeroTune computes meta-features and finds similar datasets in the knowledge base.

4. **Hyperparameter Optimization**: The system optimizes hyperparameters using the knowledge from similar datasets as guidance.

## Troubleshooting

If you encounter issues running ZeroTune:

1. **Command not found**: Make sure you're in the correct virtual environment. For Poetry 2.0+, use `poetry run zerotune` or activate the environment manually using `source /path/to/virtualenv/bin/activate`.

2. **Import errors**: Ensure the package is properly installed. Try reinstalling with `poetry install`.

3. **Metric errors**: The system uses ROC AUC as a default metric for classification tasks. Make sure your model supports probability predictions.

4. **Dataset errors**: Verify OpenML dataset IDs are correct and accessible.

## License

MIT 