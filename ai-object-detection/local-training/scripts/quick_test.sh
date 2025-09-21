#!/bin/bash
# Quick test script for the trained model

set -e

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

echo "🧪 Quick Test Script"
echo "==================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "models/train_detection_model.h5" ]; then
    echo "❌ Model not found!"
    echo "📝 Train a model first using: ./scripts/quick_train.sh"
    exit 1
fi

echo "✅ Model found!"

# Check if validation data exists
if [ -d "dataset/validation" ]; then
    validation_train_count=$(find dataset/validation/test_train -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
    validation_no_train_count=$(find dataset/validation/test_no_train -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
    
    if [ "$validation_train_count" -gt 0 ] || [ "$validation_no_train_count" -gt 0 ]; then
        echo "📊 Validation data found:"
        echo "   Test train images: $validation_train_count"
        echo "   Test no-train images: $validation_no_train_count"
        echo ""
        echo "🎯 Running validation test..."
        python -c "
from src.test_model import ModelTester
tester = ModelTester()
if tester.model:
    result = tester.validate_on_test_set()
    if result:
        print(f'\\n🎉 Validation completed!')
        print(f'Overall accuracy: {result[\"accuracy\"]:.1%}')
"
    else
        echo "📝 No validation data found. You can:"
        echo "   1. Add test images to dataset/validation/"
        echo "   2. Test individual images using the interactive tester"
    fi
else
    echo "📝 No validation directory found."
fi

echo ""
echo "🔧 For interactive testing, run:"
echo "   python src/test_model.py"
echo ""
echo "📊 Available model files:"
ls -la models/ 2>/dev/null || echo "   No models directory found"