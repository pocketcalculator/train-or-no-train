#!/bin/bash
# Setup script for Train Detection Local Development
# Run this on your Ubuntu machine to set up the environment

set -e

# Function to handle errors
handle_error() {
    echo "❌ Error occurred in setup. Check the output above for details."
    echo "💡 Try running: sudo apt install python3-venv python3-pip"
    exit 1
}

# Set error trap
trap handle_error ERR

echo "🚂 Train Detection Local Setup"
echo "=============================="

# Check if Python 3.8+ is available
if ! python3 --version >/dev/null 2>&1; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Please run this script from the train-detection-local directory."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    # Remove incomplete venv if it exists
    rm -rf venv
    
    # Create new virtual environment with --copies flag for better compatibility
    python3 -m venv venv --copies
    
    # Verify activation script exists
    if [ ! -f "venv/bin/activate" ]; then
        echo "❌ Virtual environment creation failed. Missing activate script."
        echo "💡 Try: sudo apt install python3-venv python3-pip"
        exit 1
    fi
    
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To get started:"
echo "   1. Activate the environment: source venv/bin/activate"
echo "   2. Prepare your dataset in the 'dataset' folder"
echo "   3. Run training: python src/train_model.py"
echo ""
echo "📁 Expected dataset structure:"
echo "   dataset/"
echo "   ├── train_present/    (images with trains)"
echo "   └── no_train/         (images without trains)"
echo ""
echo "💡 Recommended: 200+ images per category for best results"