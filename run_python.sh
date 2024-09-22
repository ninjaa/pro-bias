#!/bin/bash

# Get the absolute path to the project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print the updated PYTHONPATH
echo "PYTHONPATH has been updated:"
echo $PYTHONPATH

# Execute the provided command with the updated PYTHONPATH
exec "$@"