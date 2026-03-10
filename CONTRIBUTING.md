# Contributing to mne-denoise

Thank you for your interest in contributing to `mne-denoise`! This guide will help you get started with contributing code, documentation, or bug reports.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Workflow](#workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project follows the [MNE-Python Code of Conduct](https://github.com/mne-tools/mne-python/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Fork and Clone

1. **Fork** the repository on GitHub by clicking the "Fork" button.

2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-username>/mne-denoise.git
   cd mne-denoise
   ```

3. **Add the upstream remote**:

   ```bash
   git remote add upstream https://github.com/mne-tools/mne-denoise.git
   ```

## Development Environment

We recommend using a virtual environment for development.

### Using venv (recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install in editable mode with all dev dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

### Using conda

```bash
# Create environment
conda create -n mne-denoise python=3.12
conda activate mne-denoise

# Install in editable mode
pip install -e ".[dev,docs]"
pre-commit install
```

## Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
# Sync with upstream first
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/my-new-feature
# or for bug fixes:
git checkout -b fix/issue-123
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the code style guidelines (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add support for custom frequency bands in BandpassBias"
```

**Commit message guidelines:**

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep the first line under 72 characters
- Reference issues when relevant ("Fix #123: ...")

### 4. Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

### 5. Add Changelog Entry

We use [towncrier](https://towncrier.readthedocs.io/) to manage our changelog. This prevents merge conflicts and ensures standardized release notes.

When you create a Pull Request, please add a changelog entry file in `docs/changes/devel/`. The file name should be the change type (e.g., `feature.rst`, `bugfix.rst`).

For detailed instructions and available types, see [docs/changes/README.md](https://github.com/mne-tools/mne-denoise/blob/main/docs/changes/README.md).

**Author Attribution**: We encourage contributors to include their name in the changelog entry if they wish to be highlighted. In Markdown, you can link to your GitHub profile (e.g., `... (by [@YourUser](...))`).

## Code Style

We use **Ruff** for linting and formatting, configured to follow PEP 8 with NumPy docstring conventions.

### Automatic Formatting

Pre-commit hooks will automatically format your code on commit. To run manually:

```bash
# Check for linting errors
ruff check .

# Auto-fix linting errors
ruff check . --fix

# Format code
ruff format .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Docstring Style

We use NumPy-style docstrings. Example:

```python
def compute_dss(data, bias, n_components=None):
    """Compute DSS spatial filters.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data matrix.
    bias : LinearDenoiser
        Bias function that emphasizes signal of interest.
    n_components : int, optional
        Number of components to return. If None, returns all.

    Returns
    -------
    filters : ndarray, shape (n_channels, n_components)
        Spatial filters sorted by eigenvalue.
    eigenvalues : ndarray, shape (n_components,)
        Corresponding eigenvalues.

    Examples
    --------
    >>> from mne_denoise.dss import compute_dss, AverageBias
    >>> filters, eigenvalues = compute_dss(data, AverageBias())

    See Also
    --------
    DSS : Scikit-learn compatible transformer.

    References
    ----------
    .. [1] de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on
           spatial filtering. Journal of Neuroscience Methods.
    """
```

## Testing

We use **pytest** for testing. All new code should have tests.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mne_denoise --cov-report=html

# Run specific test file
pytest tests/test_linear_dss.py

# Run specific test
pytest tests/test_linear_dss.py::test_dss_epochs -v

# Run tests matching a pattern
pytest -k "zapline" -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what is being tested
- Use fixtures for common setup
- Plotting tests use the non-interactive Matplotlib backend configured in
  `tests/conftest.py`, and figures are closed automatically after each test
- Pass `show=False` when testing plotting functions
- For optional dependencies, prefer `pytest.importorskip(...)`

Example test:

```python
import numpy as np
import pytest
from mne_denoise.dss import DSS, AverageBias


@pytest.fixture
def sample_epochs():
    """Create sample epochs for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 32, 1000))


def test_dss_fit_transform_returns_correct_shape(sample_epochs):
    """Test that DSS returns sources with expected shape."""
    dss = DSS(bias=AverageBias(), n_components=5)
    sources = dss.fit_transform(sample_epochs)

    assert sources.shape[0] == 5  # n_components
    assert sources.shape[1] == sample_epochs.shape[2]  # n_times
```

### Coverage Requirements

- Aim for **100% coverage** on new code
- The CI will report coverage; check the Codecov report on your PR
- View local coverage report: `open htmlcov/index.html`

## Documentation

Documentation is built with Sphinx and hosted on GitHub Pages.

### Building Docs Locally

```bash
# Build HTML documentation
make -C docs html

# View in browser
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux
start docs/_build/html/index.html  # Windows
```

### Documentation Structure

- `docs/api.rst` - API reference (auto-generated from docstrings)
- `docs/getting-started.rst` - Installation and quick start
- `docs/dss.md` - DSS module guide
- `examples/` - Gallery examples (rendered by sphinx-gallery)

### Adding Examples

Examples are Python scripts in the `examples/` directory:

1. Create a file with prefix `plot_` (e.g., `plot_my_example.py`)
2. Follow the sphinx-gallery format with docstring headers
3. Examples are automatically built and included in the gallery

Example template:

```python
"""
Title of Example
================

Brief description of what this example demonstrates.
"""

# %%
# Section Header
# --------------
# Explanation text...

import mne_denoise

# Your code here...
```

## Submitting Changes

### Pull Request Process

1. **Push your branch** to your fork:

   ```bash
   git push origin feature/my-new-feature
   ```

2. **Open a Pull Request** on GitHub against the `main` branch.

3. **Fill out the PR template** with:
   - Description of changes
   - Related issue(s)
   - Type of change
   - Checklist items

4. **Wait for CI** to complete. All checks must pass.

5. **Address review feedback** by pushing additional commits.

6. **Squash and merge** once approved (maintainers will do this).

### PR Checklist

Before submitting, ensure:

- [ ] Code follows the project style (`ruff check .` passes)
- [ ] Code is formatted (`ruff format .` produces no changes)
- [ ] All tests pass (`pytest` exits cleanly)
- [ ] New code has tests with good coverage
- [ ] Documentation is updated if needed
- [ ] Documentation builds cleanly (`make -C docs html`)
- [ ] CHANGELOG.md is updated for user-facing changes
- [ ] Commit messages are clear and descriptive

## Issue Guidelines

### Reporting Bugs

When reporting a bug, please include:

1. **Description**: What happened vs. what you expected
2. **Reproduction steps**: Minimal code to reproduce the issue
3. **Environment**: Python version, OS, package versions
4. **Error message**: Full traceback if applicable

Use the bug report template when creating an issue.

### Requesting Features

For feature requests:

1. Check if it already exists or is planned
2. Describe the use case and motivation
3. Provide examples of how it would be used
4. Consider if you'd like to implement it yourself

## Questions?

- Open a [Discussion](https://github.com/mne-tools/mne-denoise/discussions) for questions
- Check existing issues and discussions first
- Join the [MNE-Python community](https://mne.tools/stable/overview/get_help.html)

Thank you for contributing!
