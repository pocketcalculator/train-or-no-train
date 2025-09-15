#!/bin/bash
# Quick training script with common configurations

set -e

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

echo "🚂 Quick Train Script"
echo "===================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if dataset exists
if [ ! -d "dataset/train_present" ] || [ ! -d "dataset/no_train" ]; then
    echo "❌ Dataset not found. Setting up dataset structure..."
    python src/dataset_helper.py
    echo ""
    echo "📁 Please add your images to:"
    echo "   dataset/train_present/  (images WITH trains)"
    echo "   dataset/no_train/       (images WITHOUT trains)"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Count images in dataset
train_count=$(find dataset/train_present -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
no_train_count=$(find dataset/no_train -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)

echo "📊 Dataset summary:"
echo "   Train present: $train_count images"
echo "   No train: $no_train_count images"
echo ""

# Check minimum requirements
if [ "$train_count" -lt 20 ] || [ "$no_train_count" -lt 20 ]; then
    echo "⚠️ Warning: Very small dataset (recommend 100+ images per category)"
    echo "Training may not produce good results."
    echo ""
    read -p "Continue anyway? (y/n): " continue_choice
    if [ "$continue_choice" != "y" ]; then
        echo "❌ Training cancelled"
        exit 1
    fi
fi

# Start training
echo "🚀 Starting training..."
python src/train_model.py

echo ""
echo "✅ Training script completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Review training results in models/"
echo "   2. Test your model: python src/test_model.py"
echo "   3. If accuracy is low, add more training data"