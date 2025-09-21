#!/bin/bash
# Verification script for repository structure

echo "🔍 Verifying Train Detection System Structure"
echo "=============================================="

# Function to check if directory exists and has content
check_dir() {
    local dir=$1
    local description=$2
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f | wc -l)
        echo "✅ $description: $dir ($count files)"
    else
        echo "❌ Missing: $description ($dir)"
    fi
}

# Function to check if file exists
check_file() {
    local file=$1
    local description=$2
    if [ -f "$file" ]; then
        echo "✅ $description: $file"
    else
        echo "❌ Missing: $description ($file)"
    fi
}

echo
echo "📁 Core Directories:"
check_dir "local-training" "Local ML Training"
check_dir "azure-function" "Azure Function"
check_dir "shared" "Shared Utilities"
check_dir "tests" "Testing Framework"
check_dir "docs" "Documentation"

echo
echo "🤖 Local Training Components:"
check_file "local-training/setup.sh" "Setup Script"
check_file "local-training/src/train_model.py" "Training Script"
check_file "local-training/src/test_model.py" "Testing Script"
check_file "local-training/scripts/quick_train.sh" "Quick Training"
check_dir "local-training/dataset" "Dataset Directory"

echo
echo "☁️ Azure Function Components:"
check_file "azure-function/package.json" "Node.js Config"
check_file "azure-function/deploy-to-azure.sh" "Deployment Script"
check_dir "azure-function/src" "Function Source"
check_dir "azure-function/infrastructure" "Infrastructure Code"

echo
echo "📚 Documentation:"
check_file "README.md" "Main README"
check_file "docs/ARCHITECTURE.md" "Architecture Guide"
check_file "docs/PROJECT-STATUS.md" "Status Summary"
check_file "local-training/README.md" "Training Guide"
check_file "azure-function/README.md" "Deployment Guide"

echo
echo "🎯 Structure Verification Complete!"
echo

# Check for any leftover files that might need attention
if [ -d "azure-png-processor" ]; then
    echo "⚠️  Old 'azure-png-processor' directory still exists"
fi

# Summary
total_dirs=$(find . -maxdepth 1 -type d ! -name ".*" ! -name "." | wc -l)
echo "📊 Summary: $total_dirs main directories created"
echo "🚀 Ready to begin: cd local-training && ./setup.sh"