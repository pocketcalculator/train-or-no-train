# Train Detection - Local Model Training

Build and train a custom TensorFlow model to detect trains in your railroad track images. This component runs locally on your Ubuntu machine and generates a model that can be deployed to Azure.

## 🎯 Overview

This local training environment allows you to:
- Prepare and organize your image dataset
- Train a custom CNN model optimized for your specific railroad track
- Test and validate model performance
- Export the trained model for Azure deployment

## 📁 Directory Structure

```
local-training/
├── src/
│   ├── train_model.py        # Main training script
│   ├── dataset_helper.py     # Dataset organization tools
│   └── test_model.py         # Model testing and validation
├── scripts/
│   ├── quick_train.sh        # Quick training script
│   └── quick_test.sh         # Quick testing script
├── dataset/
│   ├── train_present/        # Images WITH trains
│   ├── no_train/             # Images WITHOUT trains
│   └── validation/           # Test images for validation
├── models/                   # Generated models and results
├── venv/                     # Python virtual environment
├── requirements.txt          # Python dependencies
├── setup.sh                  # Initial setup script
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Navigate to this directory
cd local-training

# Run setup (installs Python packages)
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Prepare Your Dataset
```bash
# Create dataset structure and organize images
python src/dataset_helper.py
```

**Dataset Requirements:**
- **Minimum**: 100 images per category (train_present, no_train)
- **Recommended**: 200+ images per category
- **Formats**: JPG, PNG, JPEG
- **Consistency**: Same camera angle/position as your deployment location

### 3. Train Your Model
```bash
# Quick training (automated)
./scripts/quick_train.sh

# OR manual training (interactive)
python src/train_model.py
```

### 4. Test Your Model
```bash
# Quick validation test
./scripts/quick_test.sh

# OR interactive testing
python src/test_model.py
```

## 📊 Expected Results

### Training Performance
- **Target Accuracy**: >90% on validation set
- **Training Time**: 15-30 minutes (depends on dataset size)
- **Model Size**: ~50-100MB

### Generated Files
After training, you'll find these files in the `models/` directory:

```
models/
├── train_detection_model.h5           # Main trained model
├── best_train_detection_model.h5      # Best checkpoint during training
├── training_history.png               # Training progress plots
├── training_results.json              # Detailed training metrics
├── validation_confusion_matrix.png    # Test results visualization
└── validation_results.json            # Test performance metrics
```

## 🛠 Detailed Usage

### Dataset Preparation

#### Option 1: Manual Organization
```bash
# Create structure
mkdir -p dataset/train_present dataset/no_train

# Copy your images
cp /path/to/train/images/* dataset/train_present/
cp /path/to/empty/track/images/* dataset/no_train/
```

#### Option 2: Automated Organization
```bash
# Use the dataset helper
python src/dataset_helper.py

# Choose option 3 to organize by filename keywords
# This will automatically sort images based on filenames containing:
# - "train", "locomotive", "rail" -> train_present
# - "empty", "clear", "vacant" -> no_train
```

#### Option 3: Interactive Organization
The dataset helper provides a menu-driven interface to:
- Create proper directory structure
- Validate existing datasets
- Organize images by filename keywords
- Check for common issues

### Model Training

#### Basic Training
```bash
python src/train_model.py
```

The training script will:
1. Load and validate your dataset
2. Create a CNN model optimized for binary classification
3. Train with data augmentation and early stopping
4. Generate performance plots and metrics
5. Save the trained model

#### Training Configuration
Default settings (can be modified in the script):
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Max Epochs**: 50 (with early stopping)
- **Learning Rate**: 0.001
- **Data Split**: 80% training, 20% validation

### Model Testing

#### Quick Validation
```bash
./scripts/quick_test.sh
```

#### Comprehensive Testing
```bash
python src/test_model.py
```

Testing options:
1. **Single Image**: Test one image at a time
2. **Directory**: Test all images in a folder
3. **Validation Set**: Full test on validation data
4. **Threshold Analysis**: Find optimal confidence threshold

### Performance Optimization

#### If Accuracy is Low (<85%)
1. **Add More Data**: Increase dataset to 500+ images per category
2. **Improve Data Quality**: Remove blurry/poor quality images
3. **Balance Dataset**: Ensure roughly equal numbers in each category
4. **Add Variety**: Include different lighting, weather, train types

#### If Training is Slow
1. **Reduce Image Size**: Change from 224x224 to 128x128
2. **Increase Batch Size**: If you have enough RAM
3. **Use GPU**: Install tensorflow-gpu for CUDA acceleration

#### If Model is Too Large
1. **Reduce Model Complexity**: Fewer layers in the CNN
2. **Use Model Compression**: TensorFlow Lite conversion
3. **Quantization**: Reduce precision for smaller models

## 🔧 Troubleshooting

### Common Issues

#### "No images found in dataset"
- Check image file extensions (.jpg, .png, .jpeg)
- Verify images are in correct directories
- Run dataset validation: `python src/dataset_helper.py`

#### "Model accuracy is low"
- Add more training data (aim for 200+ per category)
- Check data quality and consistency
- Verify correct labeling of train vs no-train images
- Consider data augmentation settings

#### "Out of memory during training"
- Reduce batch size (try 16 or 8)
- Reduce image size
- Close other applications
- Consider using a cloud GPU instance

#### "Training takes too long"
- Normal training time: 15-30 minutes for 200 images
- Enable GPU acceleration if available
- Reduce max epochs if needed

### Environment Issues

#### Python Package Conflicts
```bash
# Reset virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### OpenCV Installation Issues
```bash
# Ubuntu dependencies
sudo apt update
sudo apt install python3-opencv libopencv-dev

# Reinstall opencv-python
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

#### TensorFlow GPU Setup
```bash
# Install CUDA drivers (if you have NVIDIA GPU)
# Check: https://tensorflow.org/install/gpu

# Install TensorFlow GPU
pip install tensorflow[and-cuda]==2.15.0
```

## 📈 Advanced Usage

### Custom Model Architecture
Edit `src/train_model.py` to modify the CNN architecture:
- Add/remove convolutional layers
- Change filter sizes and numbers
- Adjust dropout rates
- Modify dense layer sizes

### Data Augmentation
Current augmentation includes:
- Horizontal flipping
- Random rotation (±10%)
- Random zoom (±10%)
- Random contrast adjustment

### Transfer Learning
For better results with limited data, consider using pre-trained models:
- ResNet50, VGG16, MobileNet as base
- Fine-tune on your railroad data
- Often provides better accuracy with less data

### Hyperparameter Tuning
Key parameters to experiment with:
- Learning rate: 0.001, 0.0001, 0.01
- Batch size: 16, 32, 64
- Image size: 128x128, 224x224, 256x256
- Architecture depth and width

## 🚀 Deployment Integration

### Export for Azure
Once training is complete, the model file `train_detection_model.h5` can be:
1. Uploaded to Azure Blob Storage
2. Used in Azure Container Instances
3. Integrated with your Azure Function
4. Deployed as an Azure Machine Learning endpoint

### Model Conversion
For edge deployment or faster inference:
```bash
# Convert to TensorFlow Lite (smaller, faster)
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/train_detection_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('models/train_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
print('Model converted to TensorFlow Lite')
"
```

## 📚 Additional Resources

### Machine Learning Best Practices
- [TensorFlow Image Classification Guide](https://www.tensorflow.org/tutorials/images/classification)
- [Data Augmentation Techniques](https://www.tensorflow.org/guide/data)
- [Model Optimization](https://www.tensorflow.org/model_optimization)

### Computer Vision for Railways
- Research papers on railway monitoring
- Industrial vision system case studies
- Safety standards for automated detection

### Technical Documentation
- [TensorFlow API Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 🤝 Support

For issues or questions:
1. Check this README for common solutions
2. Validate your dataset structure
3. Review training logs for error messages
4. Test with a smaller dataset first

**Happy Training! 🚂**