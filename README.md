# ZeroTune

ZeroTune is a Python module for one-shot hyperparameter optimization using meta-learning approaches. It helps you quickly find good hyperparameters for your machine learning models based on dataset characteristics.

## Installation

### Using Poetry (Recommended)

ZeroTune uses Poetry for dependency management:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository and install
git clone https://github.com/yourusername/zerotune.git
cd zerotune
poetry install

# Activate the virtual environment
poetry shell
```

For detailed Poetry setup instructions, see [POETRY_SETUP.md](POETRY_SETUP.md).

### Using pip

You can also install using pip:

```bash
pip install -e .
```

## Requirements

- Python 3.8.1+
- Dependencies are automatically managed by Poetry

## Features

- Calculate meta-features from datasets
- Train zero-shot hyperparameter prediction models
- Run hyperparameter optimization with Optuna
- Evaluate models with random hyperparameters
- Compare performance of different HPO strategies
- **Ready-to-use pretrained models** for common algorithms
- **Visualize performance improvements** with beautiful comparison charts

## Main Functionalities

ZeroTune has two main functionalities:

1. **Offline Training**: Build a knowledge base of HPO configurations and train a bespoke ZeroTune model
2. **Inference**: Use pre-trained ZeroTune predictors to instantly suggest good hyperparameters for your dataset

## Pretrained Models

ZeroTune comes with pretrained models for common machine learning algorithms:

- **Decision Tree Classifier**: Predicts optimal values for max_depth, min_samples_split, min_samples_leaf, and max_features
- More models coming soon!

See [zerotune/models/README.md](zerotune/models/README.md) for detailed information about the pretrained models.

## Usage

### Simple Inference with Pre-trained Models

The easiest way to use ZeroTune is with pre-trained models:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from zerotune import ZeroTunePredictor, get_available_models

# See what models are available
print(get_available_models())

# Load your dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Create a predictor with the pre-trained decision tree model
predictor = ZeroTunePredictor(model_name="decision_tree")

# Get hyperparameter predictions for your dataset
hyperparams = predictor.predict(X, y)
print("Predicted hyperparameters:", hyperparams)

# Create and train a model with the predicted hyperparameters
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(**hyperparams)
model.fit(X, y)
```

### Offline Training: Building a Knowledge Base

For more advanced users, you can create your own knowledge base and train a custom ZeroTune model:

```python
from zerotune import KnowledgeBase

# Create a new knowledge base
kb = KnowledgeBase(name="my_decision_tree_kb")

# Add datasets from OpenML (classification datasets)
kb.add_dataset_from_openml(dataset_id=31)  # credit-g
kb.add_dataset_from_openml(dataset_id=40994)  # SpeedDating

# Generate synthetic datasets
kb.add_multiple_synthetic_datasets(n_datasets=100, random_seed=42)

# Compile and save the knowledge base
kb.compile_knowledge_base()
kb.save()

# Define features and target parameters for training
dataset_features = ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"]
target_params = ["params_max_depth", "params_min_samples_split", "params_min_samples_leaf", "params_max_features"]

# Train a ZeroTune model
model, score = kb.train_model(
    dataset_features=dataset_features, 
    target_params=target_params
)

print(f"Model trained with MSE: {score}")
```

### Using a Custom Trained Model

After training your own model, you can use it with the CustomZeroTunePredictor:

```python
from zerotune import CustomZeroTunePredictor

# Parameter configuration
param_config = {
    "max_depth": {
        "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999], 
        "param_type": "int", 
        "dependency": "n_samples"
    },
    "min_samples_split": {
        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
        "param_type": "float"
    },
    "min_samples_leaf": {
        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
        "param_type": "float"
    },
    "max_features": {
        "percentage_splits": [0.5, 0.7, 0.8, 0.9, 0.99], 
        "param_type": "float"
    }
}

# Create predictor with your custom model
predictor = CustomZeroTunePredictor(
    model_path="./zerotune_kb/my_decision_tree_kb/zerotune_model.joblib",
    param_config=param_config
)

# Use it to predict hyperparameters for a new dataset
hyperparams = predictor.predict(X_new, y_new)
```

### Calculate Dataset Meta-Parameters

```python
import pandas as pd
from zerotune import calculate_dataset_meta_parameters

# Load your dataset
X = pd.DataFrame(your_features)
y = pd.Series(your_target)

# Calculate meta-parameters
meta_params = calculate_dataset_meta_parameters(X, y)
print(meta_params)
```

### Run Hyperparameter Optimization with Optuna

```python
from zerotune import optuna_hpo

# Define parameter configuration for DecisionTreeClassifier
param_config = {
    "max_depth": {
        "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999], 
        "param_type": "int", 
        "dependency": "n_samples"
    },
    "min_samples_split": {
        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
        "param_type": "float"
    },
    "min_samples_leaf": {
        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
        "param_type": "float"
    },
    "max_features": {
        "percentage_splits": [0.5, 0.7, 0.8, 0.9, 0.99], 
        "param_type": "float"
    }
}

# Run Optuna HPO with ZeroTune warm-start
results = optuna_hpo(
    X, y, 
    meta_params=["n_samples", "n_features", "n_highly_target_corr"],
    param_config=param_config,
    zerotune_params=predicted_params,  # Use ZeroTune predictions as warm-start
    n_trials=100,
    n_seeds=5
)

print(f"Best performance: {results['average_best_perf']}")
print(f"Best hyperparameters: {results['all_results'][0]['best_hyperparams']}")
```

### Performance Visualization

ZeroTune can visualize the performance improvement gained from using optimized hyperparameters:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Evaluate with ZeroTune hyperparameters
model = DecisionTreeClassifier(**hyperparams, random_state=42)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)

# Compare with default hyperparameters
default_model = DecisionTreeClassifier(random_state=42)
default_scores = cross_val_score(default_model, X, y, scoring='roc_auc', cv=5)

# Calculate improvement
improvement = (np.mean(scores) - np.mean(default_scores)) / np.mean(default_scores) * 100

# Visualize results
plt.figure(figsize=(10, 6))
bars = plt.bar(['Default', 'ZeroTune'], 
             [np.mean(default_scores), np.mean(scores)],
             yerr=[np.std(default_scores), np.std(scores)],
             capsize=10, alpha=0.7, color=['lightgray', 'green'])

# Add labels and title
plt.ylabel('ROC AUC Score')
plt.title('Performance Comparison: Default vs ZeroTune Hyperparameters')
plt.ylim(0.9, 1.0)

# Add improvement annotation
plt.annotate(f'+{improvement:.2f}%', 
            xy=(1, np.mean(scores)), 
            xytext=(1.2, np.mean(scores) + 0.02),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('zerotune_performance_comparison.png')
plt.show()
```

## Running the Examples

From the repository root, after activating the Poetry environment:

```bash
# Test the pretrained model
python zerotune/examples/test_pretrained_model.py

# Run the inference example
python zerotune/examples/inference_example.py

# Run the knowledge base example
python zerotune/examples/knowledge_base_example.py

# Run the custom model example with visualization
python zerotune/examples/custom_model_example.py

# Run the advanced visualization example
python zerotune/examples/visualization_example.py

# Run the simple demonstration
python zerotune/examples/simple_example.py

# Show how Poetry simplifies package management
python zerotune/examples/poetry_example.py
```

## Running Tests

ZeroTune includes a comprehensive test suite to ensure code quality and functionality. You can run the tests using pytest:

```bash
# Install development dependencies if you haven't already
poetry install --with dev

# Run all tests
poetry run pytest

# Run tests with coverage report
poetry run pytest --cov=zerotune

# Run specific test categories
poetry run pytest tests/test_zerotune_core.py  # Core functionality tests
poetry run pytest tests/test_integration.py    # Integration tests
poetry run pytest tests/test_environment.py    # Environment validation tests

# Run tests with specific markers
poetry run pytest -m "integration"  # Run only integration tests
poetry run pytest -m "not slow"     # Skip slow tests
```

The test suite includes:

- **Unit tests**: Verify individual components function correctly
- **Integration tests**: Test complete workflows from knowledge base to prediction
- **Edge case tests**: Test behavior with empty datasets, single rows, high-dimensional data
- **Parameterized tests**: Test multiple configurations with the same test logic
- **Environment tests**: Validate Python version and dependency compatibility

For more details about the testing framework and available tests, see the [tests/README.md](tests/README.md) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Dependency Management

ZeroTune uses Poetry for dependency management. For compatible versions with Python 3.8+:

```bash
# Add a Python 3.8 compatible version of matplotlib
poetry add "matplotlib<3.10"

# Add other dependencies with specific version constraints if needed
poetry add "importlib-resources<6.0.0"
```

## Citation

If you use ZeroTune in your research or work, please cite:

```bibtex
@InProceedings{zt1_salhi_2025,
  author    = "Salhi, Tarek and Woodward, John",
  editor    = "Nicosia, Giuseppe and Ojha, Varun and Giesselbach, Sven and Pardalos, M. Panos and Umeton, Renato",
  title     = "Beyond Iterative Tuning: Zero-Shot Hyperparameter Optimisation for Decision Trees",
  booktitle = "Machine Learning, Optimization, and Data Science",
  year      = "2025",
  publisher = "Springer Nature Switzerland",
  address   = "Cham",
  pages     = "359--374",
  isbn      = "978-3-031-82481-4"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 