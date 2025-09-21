# Train Detection System - Complete Setup Guide

## 🎯 System Overview

This system extends your existing Azure PNG processor to include intelligent train detection using machine learning. It provides multiple detection methods that can work independently or together:

1. **Custom TensorFlow Model** - High accuracy for your specific use case
2. **Azure AI Services** - Cloud-based analysis with built-in models
3. **Keyword-based Fallback** - Simple but reliable backup method

## 📋 Prerequisites

### Required Software
- **Python 3.8+** (for ML components)
- **Node.js 18+** (for Azure Functions)
- **Azure CLI** (for deployment)
- **Git** (for version control)

### Azure Services Required
- **Azure Storage Account** (already configured)
- **Azure Functions** (already configured)
- **Azure AI Vision** (optional, for enhanced detection)
- **Azure Container Instances** (optional, for Python service hosting)

## 🚀 Step-by-Step Implementation

### Phase 1: Environment Setup

#### 1.1 Install Python Dependencies
```bash
# Navigate to your project
cd /home/stereo2go/code/train-or-no-train/azure-png-processor

# Create Python virtual environment
python -m venv train-detection-env
source train-detection-env/bin/activate  # Linux/Mac

# Install core packages
pip install tensorflow==2.15.0
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install pillow==10.0.0
pip install pandas==2.0.3

# Install Azure packages
pip install azure-storage-blob==12.19.0
pip install azure-ai-inference==1.0.0b1
pip install azure-core==1.29.5
pip install azure-identity==1.15.0

# Save requirements
pip freeze > requirements.txt
```

#### 1.2 Configure Environment Variables
```bash
# Create .env file for Python components
cat > .env << 'EOF'
# Azure Storage (already configured)
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string

# Train Detection Configuration
TRAIN_MODEL_PATH=./models/train_detection_model.h5
USE_CUSTOM_MODEL=true
CONFIDENCE_THRESHOLD=0.7

# Azure AI Services (optional)
AZURE_AI_ENDPOINT=your_azure_ai_endpoint
AZURE_AI_KEY=your_azure_ai_key

# Containers
INCOMING_CONTAINER_NAME=incoming
ARCHIVE_CONTAINER_NAME=archive
TRAIN_DETECTED_CONTAINER_NAME=train-detected
NO_TRAIN_CONTAINER_NAME=no-train
EOF
```

#### 1.3 Update Azure Function Configuration
Add these to your Azure Function's `local.settings.json`:
```json
{
  "Values": {
    "ENABLE_TRAIN_DETECTION": "true",
    "TRAIN_DETECTION_ENDPOINT": "http://localhost:8000",
    "CONFIDENCE_THRESHOLD": "0.7",
    "TRAIN_DETECTED_CONTAINER_NAME": "train-detected",
    "NO_TRAIN_CONTAINER_NAME": "no-train"
  }
}
```

### Phase 2: Data Preparation

#### 2.1 Organize Your Dataset
```bash
# Create dataset structure
mkdir -p dataset/{train_present,no_train,validation/{test_train,test_no_train}}

# Example structure:
# dataset/
# ├── train_present/          # 200+ images with trains
# │   ├── train_001.jpg
# │   ├── train_002.jpg
# │   └── ...
# ├── no_train/              # 200+ images without trains
# │   ├── empty_001.jpg
# │   ├── empty_002.jpg
# │   └── ...
# └── validation/            # 50+ images for final testing
#     ├── test_train/
#     └── test_no_train/
```

#### 2.2 Data Collection Guidelines
- **Minimum**: 100 images per category
- **Recommended**: 300+ images per category
- **Image quality**: Clear, focused images
- **Consistency**: Same camera angle/position as your deployment
- **Variety**: Different lighting, weather, train types
- **Balance**: Roughly equal numbers in each category

### Phase 3: Model Development

#### 3.1 Quick Evaluation with Azure AI Vision
```bash
# Test Azure AI Vision capabilities (optional)
python src/azure-vision-prototype.py
```

#### 3.2 Train Custom Model
```bash
# Train your custom model
python src/custom-train-model.py

# Follow the prompts:
# 1. Enter dataset path: ./dataset
# 2. Wait for training to complete
# 3. Review accuracy metrics
# 4. Test with sample images
```

#### 3.3 Expected Training Results
- **Target Accuracy**: >90% on validation set
- **Training Time**: 15-30 minutes (depends on dataset size)
- **Model Size**: ~50-100MB
- **Files Created**:
  - `train_detection_model.h5` (main model)
  - `training_history.png` (training plots)
  - `training_results.json` (metrics)

### Phase 4: Service Integration

#### 4.1 Test Detection Service
```bash
# Test the detection service
python src/train-detection-service.py

# Follow prompts to test:
# 1. Enter container name
# 2. Enter blob name to test
# 3. Review detection results
```

#### 4.2 Deploy Python Service (Option A: Local)
```bash
# Create simple Flask API for the detection service
cat > src/detection-api.py << 'EOF'
from flask import Flask, request, jsonify
from train_detection_service import create_train_detection_service
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize service
detection_service = create_train_detection_service()

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        container_name = data.get('container_name')
        blob_name = data.get('blob_name')
        
        result = detection_service.analyze_image_sync(container_name, blob_name)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
EOF

# Install Flask
pip install flask==3.0.0

# Run the API service
python src/detection-api.py
```

#### 4.3 Update Azure Function
```bash
# Replace the original blob monitor with enhanced version
mv src/functions/blobMonitor.ts src/functions/blobMonitor.ts.backup
mv src/functions/enhancedBlobMonitor.ts src/functions/blobMonitor.ts

# Build and test
npm run build
npm run start
```

### Phase 5: Testing & Validation

#### 5.1 End-to-End Testing
```bash
# Upload test images to incoming container
./test-scripts/upload-test-files.sh

# Monitor logs for detection results
./test-scripts/monitor-logs.sh

# Check results in different containers:
# - train-detected/ (images with trains)
# - no-train/ (images without trains)
```

#### 5.2 Performance Validation
```bash
# Test with validation dataset
python src/validate-model.py

# Expected metrics:
# - Accuracy: >90%
# - Precision: >85%
# - Recall: >85%
# - Processing time: <5 seconds per image
```

### Phase 6: Production Deployment

#### 6.1 Deploy to Azure Container Instances (Recommended)
```bash
# Create Dockerfile for Python service
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000
CMD ["python", "src/detection-api.py"]
EOF

# Build and deploy
az acr build --registry yourregistry --image train-detection:latest .
az container create --resource-group yourgroup --name train-detection \
  --image yourregistry.azurecr.io/train-detection:latest \
  --cpu 2 --memory 4 --ports 8000
```

#### 6.2 Update Function App Settings
```bash
# Set production environment variables
az functionapp config appsettings set --name yourfunctionapp \
  --resource-group yourgroup \
  --settings ENABLE_TRAIN_DETECTION=true \
             TRAIN_DETECTION_ENDPOINT=https://your-container-instance-ip:8000 \
             CONFIDENCE_THRESHOLD=0.7
```

## 🔧 Configuration Options

### Detection Methods (Priority Order)
1. **Custom Model** (`USE_CUSTOM_MODEL=true`)
   - Highest accuracy for your specific use case
   - Requires training with your data
   - ~50MB model file

2. **Azure AI Vision** (`AZURE_AI_ENDPOINT` provided)
   - Good general accuracy
   - No training required
   - Per-request pricing

3. **Keyword Analysis** (always available)
   - Basic fallback method
   - Based on filename analysis
   - Free and fast

### Performance Tuning
```bash
# Adjust confidence threshold
CONFIDENCE_THRESHOLD=0.8  # Higher = fewer false positives

# Enable/disable features
ENABLE_TRAIN_DETECTION=true
USE_CUSTOM_MODEL=true

# Container names
TRAIN_DETECTED_CONTAINER_NAME=trains-found
NO_TRAIN_CONTAINER_NAME=empty-tracks
```

## 📊 Monitoring & Troubleshooting

### Log Analysis
```bash
# Monitor Azure Function logs
func logs tail

# Check Python service logs
docker logs train-detection-container

# Analyze detection results
az storage blob list --container-name train-detected --account-name youraccount
```

### Common Issues
1. **Low accuracy**: Retrain with more diverse data
2. **Slow processing**: Optimize image size, use faster model
3. **Memory issues**: Reduce batch size, optimize containers
4. **False positives**: Increase confidence threshold
5. **False negatives**: Lower confidence threshold, improve training data

### Performance Metrics
- **Processing Speed**: Target <5 seconds per image
- **Accuracy**: Target >90% overall
- **Throughput**: Target 100+ images per hour
- **Availability**: Target 99%+ uptime

## 🎯 Success Criteria

### Technical Validation
- [ ] Model accuracy >90% on validation set
- [ ] Processing time <5 seconds per image
- [ ] Azure Function integration working
- [ ] All containers created and accessible
- [ ] End-to-end pipeline functional

### Business Validation
- [ ] Correctly identifies trains in your specific environment
- [ ] False positive rate <10%
- [ ] False negative rate <10%
- [ ] System runs reliably for 24+ hours
- [ ] Easy to monitor and maintain

## 🔄 Maintenance & Updates

### Regular Tasks
- **Weekly**: Review detection accuracy, check logs
- **Monthly**: Retrain model with new data if needed
- **Quarterly**: Update dependencies, review performance

### Model Improvement
1. Collect misclassified images
2. Add to training dataset
3. Retrain model
4. Deploy updated model
5. Monitor improved performance

## 📚 Next Steps

1. **Enhanced Features**:
   - Train type classification (freight vs passenger)
   - Direction detection (approaching vs departing)
   - Speed estimation
   - Integration with external systems

2. **Scalability**:
   - Batch processing optimization
   - Multi-region deployment
   - Real-time streaming processing

3. **Analytics**:
   - Historical trend analysis
   - Automated reporting
   - Predictive maintenance alerts

## 🆘 Support & Resources

- **Documentation**: Check README.md files in each directory
- **Troubleshooting**: Run `./validate-best-practices.sh`
- **Performance**: Monitor Application Insights
- **Updates**: Follow TensorFlow and Azure updates

---

**Status**: Ready for Implementation  
**Estimated Setup Time**: 4-6 hours  
**Prerequisites**: Azure account, dataset ready  
**Next Action**: Start with Phase 1 - Environment Setup