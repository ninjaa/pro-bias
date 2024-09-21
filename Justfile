# Justfile

# Set default recipe to run when no arguments are provided
default:
    @echo "Specify a command to run: lint, test, or clean"

# Install dependencies
deps:
    pip install -r requirements.txt

# Run linters using Ruff
lint:
    ruff .

# Run unit tests with pytest
test:
    pytest

# Clean up pyc files and __pycache__ directories
clean:
    find . -name '*.pyc' -delete
    find . -type d -name '__pycache__' -exec rm -rf {} +
