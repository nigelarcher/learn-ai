#!/bin/bash

# Script to run the transformer demo

echo "Health Insurance Claims Transformer Demo"
echo "======================================="
echo ""
echo "This demo shows a transformer built from scratch (no libraries)"
echo "for processing health insurance claims."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    
    echo "Installing numpy (the only dependency)..."
    "$VENV_PATH/bin/pip" install numpy
fi

# Navigate to src directory and run the demo
cd "$SCRIPT_DIR/src"

echo "Running transformer demo..."
echo ""
"$VENV_PATH/bin/python" example_usage.py

echo ""
echo "Demo complete! Check the docs/ directory for detailed explanations."