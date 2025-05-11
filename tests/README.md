# Vigilance System Tests

This directory contains tests for the Vigilance System.

## Running Tests

To run all tests:

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows

# Run tests
pytest
```

To run a specific test file:

```bash
pytest tests/test_config.py
```

To run a specific test:

```bash
pytest tests/test_config.py::test_config_singleton
```

## Test Coverage

To generate a test coverage report:

```bash
# Install coverage package
pip install pytest-cov

# Run tests with coverage
pytest --cov=vigilance_system

# Generate HTML report
pytest --cov=vigilance_system --cov-report=html
```

The HTML report will be generated in the `htmlcov` directory.

## Writing Tests

When writing tests, follow these guidelines:

1. Create a new test file for each module being tested
2. Use descriptive test names that explain what is being tested
3. Use pytest fixtures for common setup
4. Mock external dependencies (e.g., cameras, network, etc.)
5. Test both success and failure cases
