#!/bin/bash
# Model Comparison Script
# Compare performance between different model versions

set -e

echo "📊 Model Performance Comparison"
echo "==============================="

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Check for models directory
if [ ! -d "models" ]; then
    echo "❌ Models directory not found!"
    exit 1
fi

# List available model versions
echo "📁 Available model versions:"
echo ""

# Find all training results
results_files=$(find models -name "training_results*.json" | sort)

if [ -z "$results_files" ]; then
    echo "❌ No training results found!"
    exit 1
fi

# Create comparison table
echo "| Version | Date | Accuracy | Precision | Recall | Dataset Size |"
echo "|---------|------|----------|-----------|--------|--------------|"

best_accuracy=0
best_version=""

for results_file in $results_files; do
    if [ -f "$results_file" ]; then
        # Extract version from filename
        version=$(basename "$results_file" | sed 's/training_results_//' | sed 's/.json//')
        if [ "$version" = "training_results" ]; then
            version="current"
        fi
        
        # Parse JSON data (basic extraction - requires jq for complex parsing)
        if command -v jq >/dev/null 2>&1; then
            # Use jq if available
            accuracy=$(jq -r '.validation.accuracy // .validation.accuracy // 0' "$results_file")
            precision=$(jq -r '.validation.precision // .validation.precision // 0' "$results_file")
            recall=$(jq -r '.validation.recall // .validation.recall // 0' "$results_file")
            dataset_size=$(jq -r '.dataset_size.total // .epochs_trained // "N/A"' "$results_file")
            timestamp=$(jq -r '.timestamp // "N/A"' "$results_file")
            
            # Format numbers
            accuracy_fmt=$(printf "%.3f" "$accuracy")
            precision_fmt=$(printf "%.3f" "$precision")
            recall_fmt=$(printf "%.3f" "$recall")
            
            # Track best model
            if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$accuracy
                best_version=$version
            fi
            
        else
            # Fallback without jq
            accuracy_fmt="N/A"
            precision_fmt="N/A"
            recall_fmt="N/A"
            dataset_size="N/A"
            timestamp="N/A"
        fi
        
        # Format date
        if [ "$timestamp" != "N/A" ]; then
            date_fmt=$(echo "$timestamp" | cut -d'T' -f1)
        else
            date_fmt="N/A"
        fi
        
        echo "| $version | $date_fmt | $accuracy_fmt | $precision_fmt | $recall_fmt | $dataset_size |"
    fi
done

echo ""

if [ -n "$best_version" ]; then
    echo "🏆 Best performing model: $best_version (accuracy: $best_accuracy)"
else
    echo "ℹ️ Install 'jq' for detailed comparison: sudo apt install jq"
fi

echo ""
echo "📊 Available model files:"
ls -la models/*.h5 2>/dev/null | awk '{print "   " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}'

echo ""
echo "📈 Available training plots:"
ls -la models/*.png 2>/dev/null | awk '{print "   " $9}'

echo ""
echo "💡 Tips for model improvement:"
echo "   1. Add more diverse training data"
echo "   2. Balance your dataset (equal classes)"
echo "   3. Include challenging edge cases"
echo "   4. Use lower learning rates for fine-tuning"
echo "   5. Monitor validation accuracy to avoid overfitting"

# Offer to view specific model details
echo ""
read -p "View detailed results for a specific version? (version name or 'n'): " selected_version

if [ "$selected_version" != "n" ] && [ "$selected_version" != "" ]; then
    results_file="models/training_results_$selected_version.json"
    if [ "$selected_version" = "current" ]; then
        results_file="models/training_results.json"
    fi
    
    if [ -f "$results_file" ]; then
        echo ""
        echo "📋 Detailed results for $selected_version:"
        echo "----------------------------------------"
        if command -v jq >/dev/null 2>&1; then
            jq '.' "$results_file"
        else
            cat "$results_file"
        fi
    else
        echo "❌ Results file not found: $results_file"
    fi
fi