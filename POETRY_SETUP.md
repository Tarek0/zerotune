# Setting Up ZeroTune with Poetry

This guide will help you set up ZeroTune using Poetry for dependency management.

## Requirements

- Python 3.8.1+
- [Poetry](https://python-poetry.org/docs/#installation)

## Installation Steps

### 1. Install Poetry (if you haven't already)

For macOS/Linux/WSL:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

For Windows (PowerShell):
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Add Poetry to your PATH if the installer didn't do it automatically.

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/zerotune.git
cd zerotune
```

### 3. Set Up the Project with Poetry

Install dependencies:
```bash
poetry install
```

This will create a virtual environment and install all the required dependencies.

### 4. Activate the Virtual Environment

```bash
poetry shell
```

### 5. Run the Examples

```bash
# Run the inference example
python zerotune/examples/inference_example.py

# Run the knowledge base example
python zerotune/examples/knowledge_base_example.py

# Run the Poetry example
python zerotune/examples/poetry_example.py
```

## Troubleshooting

### Python Version Issues

If you see an error like:
```
The current project's supported Python range (>=3.8.1,<4.0) is not compatible with your Python version
```

Make sure you're using Python 3.8.1 or newer. You can check your Python version with:

```bash
python --version
```

If you have multiple Python versions installed, you can specify which one Poetry should use:

```bash
poetry env use /path/to/python3.8
```

or with pyenv:

```bash
pyenv local 3.8.x
poetry install
```

## Development Workflow

### Adding New Dependencies

```bash
# Add a production dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name
```

### Updating Dependencies

```bash
poetry update
```

### Running Tests (when added)

```bash
poetry run pytest
```

### Building the Package

```bash
poetry build
```

This will generate distribution files in the `dist` directory.

### Publishing to PyPI (if needed)

```bash
poetry publish
```

## Poetry Commands Reference

- **Install all dependencies**: `poetry install`
- **Update dependencies**: `poetry update`
- **Add a dependency**: `poetry add package-name`
- **Remove a dependency**: `poetry remove package-name`
- **Activate virtual environment**: `poetry shell`
- **Run a command in the virtual environment**: `poetry run command`
- **Show installed packages**: `poetry show`
- **Build the package**: `poetry build`
- **Publish to PyPI**: `poetry publish`

For more information, refer to the [Poetry documentation](https://python-poetry.org/docs/). 