#!/bin/bash

# This script sets up a Python virtual environment, installs dependencies,
# and runs the image processing script.

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- Setup Virtual Environment ---
# If the venv directory exists but is incomplete, remove it
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Incomplete virtual environment found. Removing to recreate."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Please ensure python3 and the 'venv' module are installed."
        exit 1
    fi
fi

# --- Activate Virtual Environment and Install Dependencies ---
echo "Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"

# Install required packages
pip install -q Pillow piexif
if [ $? -ne 0 ]; then
    echo "Error: Failed to install required Python packages."
    deactivate
    exit 1
fi

echo "Dependencies are up to date."
echo ""

# --- Run the Image Processing Script ---
echo "Running the image processing script..."
echo "========================================"

# Run the python script, passing along any arguments you gave this shell script
python3 "$SCRIPT_DIR/process_image.py" "$@"

echo "========================================"
echo "Script execution finished."

# --- Deactivate Virtual Environment ---
deactivate
echo "Virtual environment deactivated."
