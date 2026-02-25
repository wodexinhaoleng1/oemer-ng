# Contributing to oemer-ng

Thank you for your interest in contributing to oemer-ng! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/oemer-ng.git
   cd oemer-ng
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Git

### Install Development Dependencies

```bash
pip install -e ".[dev,onnx,training]"
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 100 characters (configured in pyproject.toml)
- Use Black for code formatting
- Use type hints where appropriate

### Formatting Code

```bash
# Format all code
black src/ tests/ examples/

# Check formatting without changes
black --check src/ tests/ examples/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=oemer_ng --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::test_omr_model_forward
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive names
- Test both normal and edge cases

Example:
```python
def test_model_forward():
    """Test model forward pass with valid input."""
    model = OMRModel(num_classes=128)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 128)
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

### PR Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Add tests for new features
- Update documentation as needed
- Keep changes focused and minimal

## Code Organization

```
oemer-ng/
├── src/oemer_ng/          # Main package
│   ├── models/            # Neural network architectures
│   ├── inference/         # Inference pipeline
│   ├── training/          # Training utilities
│   ├── quantization/      # Model quantization
│   ├── export/            # Model export (GGML, ONNX)
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── examples/              # Example scripts
└── docs/                  # Documentation (future)
```

## Adding New Features

### New Model Architecture

1. Add model in `src/oemer_ng/models/`
2. Export from `models/__init__.py`
3. Add tests in `tests/test_model.py`
4. Add example usage in `examples/`
5. Update README.md

### New Export Format

1. Add exporter in `src/oemer_ng/export/`
2. Update `export/__init__.py`
3. Add tests
4. Update examples and documentation

## Documentation

- Use Google-style docstrings
- Document all public APIs
- Include examples in docstrings where helpful
- Update README.md for user-facing changes

Example docstring:
```python
def preprocess(self, image: np.ndarray) -> torch.Tensor:
    """
    Preprocess image for inference.
    
    Args:
        image: Input image as numpy array (H, W, C)
        
    Returns:
        Preprocessed tensor ready for model
        
    Example:
        >>> prep = ImagePreprocessor()
        >>> img = np.random.randint(0, 255, (512, 512, 3))
        >>> tensor = prep.preprocess(img)
        >>> tensor.shape
        torch.Size([3, 512, 512])
    """
```

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/stack traces

## Questions?

- Open an issue for questions
- Check existing issues first
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to oemer-ng! 🎵
