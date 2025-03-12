# Contributing to ZeroTune

Thank you for your interest in contributing to ZeroTune! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

There are several ways to contribute to ZeroTune:

1. **Report bugs**: Open an issue describing the bug, steps to reproduce, and expected behavior.
2. **Request features**: Open an issue describing the new feature and its potential benefits.
3. **Submit pull requests**: Implement new features or fix bugs through PRs.
4. **Improve documentation**: Help us improve our docs, examples, or comments.
5. **Share examples**: Create examples that demonstrate ZeroTune's capabilities.

## Development Setup

1. **Fork the repository**:
   ```bash
   git clone https://github.com/yourusername/zerotune.git
   cd zerotune
   ```

2. **Install dependencies with Poetry**:
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Python Version Compatibility

ZeroTune supports Python 3.8.1 and higher. When adding new dependencies, ensure they're compatible with Python 3.8.1+:

```bash
# Example: Adding a Python 3.8 compatible matplotlib version
poetry add "matplotlib<3.10"
```

## Pull Request Process

1. **Create a branch**: Work on a feature branch, not directly on main.
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**: Implement your feature or fix.

3. **Run tests**: Ensure all tests pass.
   ```bash
   poetry run pytest
   ```

4. **Update documentation**: If necessary, update the README or other docs.

5. **Submit your PR**: Create a pull request with a clear description of the changes.

## Testing

ZeroTune uses pytest for testing. Make sure to add tests for any new features:

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=zerotune
```

## Style Guide

ZeroTune follows these style conventions:

- Code is formatted with Black (line length 88)
- Imports are sorted with isort
- Type hints are used where possible
- Docstrings follow the NumPy/SciPy style

You can run the formatters with:
```bash
poetry run black zerotune
poetry run isort zerotune
```

## License

By contributing to ZeroTune, you agree that your contributions will be licensed under the project's MIT License. 