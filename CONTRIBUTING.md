# Contributing to SOLIS

Thank you for your interest in contributing to SOLIS! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

## Code of Conduct

This project follows standard open-source community guidelines:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Detailed steps to reproduce the problem
- Expected vs. actual behavior
- Screenshots (if applicable)
- Your environment (OS, Python version, SOLIS version)
- Error messages or logs

Use the bug report template when creating issues.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- Detailed description of the proposed feature
- Explanation of why this enhancement would be useful
- Possible implementation approaches (if you have ideas)

Use the feature request template when creating issues.

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.13
- Git
- Virtual environment tool

### Setup Steps

1. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/SOLIS.git
   cd SOLIS
   ```

2. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/el-bastos/SOLIS.git
   ```

3. **Create virtual environment:**
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-qt pytest-cov flake8
   ```

5. **Install development dependencies:**
   ```bash
   pip install pyinstaller  # For building
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Maximum line length: 127 characters
- Use docstrings for all public functions and classes

### Backend Data Structures

**IMPORTANT:** Read `docs/BACKEND_REFERENCE.md` before making changes to backend code.

- **NEVER use dictionaries for backend data** - use dataclasses exclusively
- Backend uses dataclasses: `KineticsResult`, `FitParameters`, etc.
- This ensures type safety and IDE autocomplete

### Code Organization

```
SOLIS/
├── core/              # Analysis engine (no GUI dependencies)
├── heterogeneous/     # Heterogeneous analysis methods
├── gui/              # GUI components (PyQt6)
├── plotting/         # Visualization modules
├── utils/            # Utilities and helpers
└── tests/            # Test files
```

### Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

Example:
```python
import os
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QWidget

from core.kinetics_fitter import KineticsFitter
from utils.logger_config import setup_logger
```

### Documentation

- Use Google-style docstrings
- Include type hints for function parameters and return values
- Document all public APIs

Example:
```python
def fit_kinetics(data: np.ndarray, model: str = "single") -> KineticsResult:
    """
    Fit kinetics data to specified model.

    Args:
        data: Array of time-resolved luminescence data (shape: [n, 2])
        model: Model type - "single" or "biexponential"

    Returns:
        KineticsResult containing fitted parameters and statistics

    Raises:
        ValueError: If data shape is invalid
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_kinetics.py
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

Example:
```python
def test_single_exponential_fit_returns_correct_lifetime():
    """Test that single exponential fit returns expected lifetime."""
    # Arrange
    data = generate_test_data(tau=50.0)

    # Act
    result = fit_kinetics(data, model="single")

    # Assert
    assert abs(result.tau - 50.0) < 1.0  # Within 1 us tolerance
```

## Submitting Changes

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(heterogeneous): add vesicle size distribution analysis

Implements polydisperse vesicle analysis using gamma distribution.
Includes parameter estimation and visualization.

Closes #123
```

```
fix(gui): resolve crash when loading empty data files

Handle edge case where CSV files are empty or contain only headers.
Shows user-friendly error message instead of crashing.

Fixes #456
```

### Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Run tests locally** before submitting
4. **Update CHANGELOG** (if applicable)
5. **Keep PR focused** - one feature or fix per PR
6. **Link related issues** in PR description
7. **Respond to review comments** promptly

### Review Process

- Maintainers will review your PR
- Address requested changes
- Once approved, PR will be merged
- Delete your feature branch after merge

## Project-Specific Guidelines

### Before All Coding

1. **Read `docs/memory.md`** for project context and current status
2. **Read `docs/BACKEND_REFERENCE.md`** for backend architecture
3. **Ask before coding** if you're unsure about approach
4. **Commit and backup frequently**

### Key Principles

- **Don't overcomplicate** - keep solutions simple and maintainable
- **Be assertive** - propose clear solutions
- **Never use dictionaries for backend data** - use dataclasses
- **Test with real data** when possible

### GUI Changes

- Use PyQt6 for all GUI components
- Follow Material Design principles where applicable
- Test on multiple screen resolutions
- Ensure keyboard navigation works
- Add tooltips for non-obvious controls

### Performance Considerations

- Use Numba JIT for computationally intensive functions
- Profile before optimizing
- Document performance-critical code
- Include timing benchmarks for major optimizations

### Data Safety

- Never modify user data files
- Always create backups before destructive operations
- Handle errors gracefully with user-friendly messages
- Validate all user inputs

## Documentation

### When to Update Documentation

- When adding new features
- When changing existing behavior
- When fixing bugs that affect documented behavior
- When deprecating features

### Documentation Locations

- **User Guide:** `docs/USER_GUIDE.md`
- **API Documentation:** Docstrings in code
- **Development Docs:** `docs/BACKEND_REFERENCE.md`, `docs/memory.md`
- **Build Instructions:** `BUILD.md`
- **README:** General overview and quick start

## Getting Help

If you need help:

1. Check existing documentation
2. Search closed issues
3. Ask in issue comments
4. Contact maintainer: elbastos@iq.usp.br

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Code comments for specific implementations

## License

By contributing to SOLIS, you agree that your contributions will be licensed under the CC BY-NC 4.0 license.

---

Thank you for contributing to SOLIS! Your help makes this project better for everyone.
