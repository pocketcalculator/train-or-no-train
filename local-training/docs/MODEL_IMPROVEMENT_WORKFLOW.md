# Model Improvement Workflow

## Overview
This guide provides a systematic approach to improving your train detection model over time as you acquire more images.

## Quick Start

### 1. Add New Images
```bash
# Add new images to appropriate folders
cp new_train_images/* dataset/train_present/
cp new_no_train_images/* dataset/no_train/
```

### 2. Incremental Training
```bash
# Simple retraining with new data
./scripts/incremental_train.sh

# With custom settings
./scripts/incremental_train.sh --learning-rate 0.00005 --version v2.1
```

### 3. Compare Results
```bash
# View performance comparison
./scripts/compare_models.sh
```

## Detailed Workflow

### Phase 1: Data Collection (Weekly)
1. **Identify Gaps**: Review failed predictions from `test_model.py`
2. **Target Collection**: Focus on underrepresented scenarios
3. **Quality Check**: Ensure new images meet quality standards
4. **Organization**: Follow naming conventions and folder structure

### Phase 2: Incremental Training (Bi-weekly)
1. **Backup**: Script automatically backs up current model
2. **Fine-tune**: Use reduced learning rate (0.0001 vs 0.001)
3. **Version**: Tag each training session
4. **Validate**: Test on held-out validation set

### Phase 3: Evaluation (After each training)
1. **Compare**: Use comparison script to track progress
2. **Test**: Run comprehensive testing on diverse images
3. **Document**: Record observations and improvements needed

## Best Practices

### Data Collection Strategy
- **Start Small**: Add 20-50 images at a time
- **Balance First**: Focus on balancing classes before adding volume
- **Quality over Quantity**: Better to have 100 good images than 500 poor ones
- **Edge Cases**: Prioritize scenarios where current model fails

### Training Strategy
- **Learning Rate**: Start with 0.0001 for fine-tuning existing models
- **Epochs**: Use 20-30 epochs for incremental training
- **Early Stopping**: Let the model decide when to stop
- **Validation**: Always reserve 20% for validation

### Model Management
- **Versioning**: Use descriptive version tags (v1.0, v1.1_balanced_data)
- **Backups**: Automatic backup before each training session
- **Documentation**: Keep notes on what changed between versions

## Performance Tracking

### Key Metrics to Monitor
1. **Validation Accuracy**: Primary metric for model quality
2. **Precision**: How many predicted trains are actually trains
3. **Recall**: How many actual trains are correctly identified
4. **Class Balance**: Dataset distribution between classes

### Warning Signs
- **Overfitting**: Training accuracy >> Validation accuracy
- **Plateau**: No improvement after adding more data
- **Class Bias**: High accuracy on one class, poor on another

## Troubleshooting

### Model Not Improving?
1. **Data Quality**: Check image quality and labeling accuracy
2. **Data Diversity**: Add more varied scenarios
3. **Class Balance**: Ensure roughly equal examples per class
4. **Learning Rate**: Try different rates (0.00001 - 0.001)

### Poor Performance on Specific Scenarios?
1. **Targeted Collection**: Collect more examples of problem scenarios
2. **Data Augmentation**: Review if augmentation helps or hurts
3. **Hard Negative Mining**: Focus on images model gets wrong

### Model Too Large/Slow?
1. **Model Pruning**: Remove unnecessary weights
2. **Quantization**: Reduce model precision
3. **Architecture Changes**: Consider smaller model variants

## Advanced Techniques

### Transfer Learning
- Start with pre-trained models (ResNet, EfficientNet)
- Fine-tune on your specific data
- Requires model architecture changes

### Data Augmentation Optimization
- Analyze which augmentations help vs hurt
- Adjust augmentation parameters based on data
- Consider domain-specific augmentations

### Active Learning
- Use model uncertainty to guide data collection
- Focus on images where model is least confident
- Iteratively improve on weak points

## File Structure for Versioning
```
models/
├── train_detection_model.h5              # Current best model
├── best_train_detection_model.h5          # Best from last training
├── training_results.json                 # Current results
├── backups/                              # Model backups
│   ├── model_backup_20250915_143022.h5
│   └── results_backup_20250915_143022.json
├── train_detection_model_v1.0.h5         # Version-specific models
├── training_results_v1.0.json            # Version-specific results
└── training_history_v1.0.png             # Version-specific plots
```

## Automation Scripts

### Available Scripts
- `./scripts/incremental_train.sh` - Main training workflow
- `./scripts/compare_models.sh` - Performance comparison
- `./scripts/quick_train.sh --force` - Full retraining
- `./scripts/quick_test.sh` - Quick model testing

### Script Parameters
```bash
# Incremental training options
./scripts/incremental_train.sh \
    --learning-rate 0.00005 \
    --version v2.1_improved_lighting \
    --force

# Custom epochs and dataset
python src/train_model_incremental.py \
    --epochs 50 \
    --learning_rate 0.0001 \
    --version_tag v2.2 \
    --dataset_path custom_dataset/
```

## Success Metrics

### Short-term Goals (1-2 weeks)
- [ ] Balanced dataset (1:1 or 2:1 ratio)
- [ ] 90%+ validation accuracy
- [ ] Good performance on your specific use cases

### Medium-term Goals (1-2 months)
- [ ] 500+ high-quality images
- [ ] Robust performance across weather/lighting
- [ ] Low false positive rate

### Long-term Goals (3-6 months)
- [ ] 1000+ diverse images
- [ ] Production-ready performance
- [ ] Integrated with deployment pipeline

## Next Steps
1. Follow data collection guide: `docs/DATA_COLLECTION_GUIDE.md`
2. Run your first incremental training session
3. Set up regular improvement cycles
4. Monitor and document progress