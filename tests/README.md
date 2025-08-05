# ZeroTune Test Suite

This directory contains the test suite for the ZeroTune package. The test files follow a modular structure that mirrors the organization of the package itself.

## Test Files

- **test_zerotune.py**: Tests for the main `ZeroTune` class and API, covering initialization, functionality, and end-to-end workflows.
- **test_zerotune_cli.py**: Tests for the command-line interface functionality.
- **test_data_loading.py**: Tests for the data loading module, including loading datasets from OpenML and preparing data.
- **test_feature_extraction.py**: Tests for the feature extraction module, covering dataset meta-parameter calculation.
- **test_knowledge_base.py**: Tests for the knowledge base module, covering saving, loading, and finding similar datasets.
- **test_model_configs.py**: Tests for the model configuration module, including parameter settings for different algorithms.
- **test_optimization.py**: Tests for the optimization module, covering multi-seed hyperparameter optimization with Optuna.
- **test_predictor_training.py**: Tests for the advanced predictor training module, including RFECV feature selection, NMAE/Top-K evaluation metrics, and multi-seed training data processing.
- **test_predictors.py**: Tests for the `ZeroTunePredictor` class, covering zero-shot hyperparameter prediction and model loading.

## Running Tests

To run the test suite, you need to install the required dependencies. You can do this using:

```bash
# Install ZeroTune in development mode
pip install -e .

# Install test dependencies
pip install -r requirements-test.txt
```

Once the dependencies are installed, you can run the entire test suite with:

```bash
pytest
```

Or run specific tests or test files with:

```bash
# Run a specific test file
pytest tests/test_zerotune.py

# Run a specific test function
pytest tests/test_zerotune.py::test_zerotune_initialization

# Run with verbose output
pytest -v
```

## Test Coverage

To check test coverage, you can use:

```bash
pytest --cov=zerotune
```

## Mocking

Many tests use mocking to avoid relying on external services or databases. This makes the tests more reliable and faster to run.

For example, OpenML data fetching is mocked to avoid making actual API calls during testing.

## Test Fixtures

The test suite uses pytest fixtures to set up common test objects, such as synthetic datasets and model configurations. These fixtures help reduce code duplication and make tests more focused and readable.
