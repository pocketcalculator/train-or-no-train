#!/bin/bash
# Incremental Model Improvement Script
# This script helps you retrain your model with new data while preserving learned features

set -e

echo "🔄 Train Detection Model - Incremental Training"
echo "==============================================="

# Parse command line arguments
FORCE_RETRAIN=false
VERSION_TAG=""
LEARNING_RATE="0.0001"  # Lower learning rate for fine-tuning

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_RETRAIN=true
            shift
            ;;
        --version)
            VERSION_TAG="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force] [--version <tag>] [--learning-rate <rate>]"
            exit 1
            ;;
    esac
done

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Count current dataset
train_count=$(find dataset/train_present -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
no_train_count=$(find dataset/no_train -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
total_count=$((train_count + no_train_count))

echo "📊 Current Dataset:"
echo "   - Train present: $train_count images"
echo "   - No train: $no_train_count images"
echo "   - Total: $total_count images"
echo ""

# Check if we have enough new data to justify retraining
if [ "$total_count" -lt 50 ] && [ "$FORCE_RETRAIN" = false ]; then
    echo "⚠️ Small dataset detected ($total_count images)"
    echo "💡 Recommendations:"
    echo "   - Add more images (target: 100+ total)"
    echo "   - Focus on balancing classes (current ratio: $train_count:$no_train_count)"
    echo ""
    read -p "Continue with current dataset? (y/n): " continue_choice
    if [ "$continue_choice" != "y" ]; then
        echo "❌ Training cancelled"
        echo ""
        echo "📋 Next steps:"
        echo "   1. Add more images to dataset/"
        echo "   2. Use --force flag to skip this check"
        exit 1
    fi
fi

# Check for existing model
EXISTING_MODEL=""
if [ -f "models/train_detection_model.h5" ]; then
    EXISTING_MODEL="models/train_detection_model.h5"
    echo "📂 Found existing model: $EXISTING_MODEL"
elif [ -f "models/best_train_detection_model.h5" ]; then
    EXISTING_MODEL="models/best_train_detection_model.h5"
    echo "📂 Found existing model: $EXISTING_MODEL"
else
    echo "ℹ️ No existing model found - will train from scratch"
fi

# Create backup of current model if it exists
if [ -n "$EXISTING_MODEL" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="models/backups"
    mkdir -p "$BACKUP_DIR"
    
    echo "💾 Creating backup of current model..."
    cp "$EXISTING_MODEL" "$BACKUP_DIR/model_backup_$TIMESTAMP.h5"
    
    if [ -f "models/training_results.json" ]; then
        cp "models/training_results.json" "$BACKUP_DIR/results_backup_$TIMESTAMP.json"
    fi
    
    echo "✅ Backup created: $BACKUP_DIR/model_backup_$TIMESTAMP.h5"
fi

# Version tagging
if [ -n "$VERSION_TAG" ]; then
    echo "🏷️ Version tag: $VERSION_TAG"
else
    VERSION_TAG="v$(date +%Y%m%d_%H%M%S)"
    echo "🏷️ Auto-generated version: $VERSION_TAG"
fi

# Start incremental training
echo ""
echo "🚀 Starting incremental training..."
echo "⚙️ Configuration:"
echo "   - Learning rate: $LEARNING_RATE (reduced for fine-tuning)"
echo "   - Version: $VERSION_TAG"
echo "   - Existing model: ${EXISTING_MODEL:-'None (training from scratch)'}"

# Create incremental training script call
python src/train_model_incremental.py \
    --learning_rate "$LEARNING_RATE" \
    --version_tag "$VERSION_TAG" \
    --existing_model "$EXISTING_MODEL"

# Save training metadata
echo "📝 Saving training metadata..."
cat > "models/training_metadata_$VERSION_TAG.json" << EOF
{
    "version": "$VERSION_TAG",
    "timestamp": "$(date -Iseconds)",
    "dataset_size": {
        "train_present": $train_count,
        "no_train": $no_train_count,
        "total": $total_count
    },
    "training_config": {
        "learning_rate": "$LEARNING_RATE",
        "existing_model": "$EXISTING_MODEL",
        "backup_created": true
    }
}
EOF

echo ""
echo "✅ Incremental training completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Test the improved model: python src/test_model.py"
echo "   2. Compare with previous version: ./scripts/compare_models.sh"
echo "   3. Add more training data and repeat"
echo ""
echo "💾 Files created:"
echo "   - Updated model: models/train_detection_model.h5"
echo "   - Training metadata: models/training_metadata_$VERSION_TAG.json"
echo "   - Model backup: models/backups/model_backup_*.h5"