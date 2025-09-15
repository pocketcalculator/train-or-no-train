# Azure PNG Processor - Train Detection System

A complete train detection system combining Azure cloud services with local machine learning for real-time monitoring of railroad tracks through image processing.

## 🎯 System Overview

This project provides an end-to-end solution for:

- **Real-time Image Monitoring**: Azure Functions monitor blob storage for new images
- **AI-Powered Detection**: Custom TensorFlow models detect trains in railroad track images  
- **Local Model Training**: Build and optimize detection models on your own hardware
- **Cloud Integration**: Deploy trained models to Azure for scalable inference
- **Automated Processing**: Seamless workflow from image upload to detection results

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Image Upload  │───▶│  Azure Storage  │───▶│ Azure Functions │
│   (Railroad     │    │   (Blob Store)  │    │ (Blob Monitor)  │
│    Cameras)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Detection     │◀───│  Train Detection│
                       │   Results       │    │     Model       │
                       │   (Database)    │    │   (TensorFlow)  │
                       └─────────────────┘    └─────────────────┘
                                                        ▲
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Model Export  │───▶│  Local Training │
                       │   to Azure      │    │   Environment   │
                       │                 │    │   (Ubuntu PC)   │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
azure-png-processor/
├── 🌩️ Azure Functions (Cloud Processing)
│   ├── src/functions/
│   │   └── enhancedBlobMonitor.ts    # Main Azure Function
│   ├── infrastructure/
│   │   ├── main.bicep                # Azure infrastructure
│   │   └── main.bicepparam           # Deployment parameters
│   └── package.json                  # Node.js dependencies
│
└── 🤖 Local ML Training
    └── train-detection-local/
        ├── src/
        │   ├── train_model.py        # Model training script
        │   ├── test_model.py         # Model validation
        │   └── dataset_helper.py     # Dataset management
        ├── scripts/
        │   ├── quick_train.sh        # Automated training
        │   └── quick_test.sh         # Quick validation
        ├── dataset/                  # Training images
        └── models/                   # Trained models
```

## 🚀 Quick Start

### 1. Local Model Training (First Step)

Train your custom train detection model locally:

```bash
# Navigate to training environment
cd train-detection-local

# Set up Python environment
./setup.sh

# Prepare your image dataset
python src/dataset_helper.py

# Train the model
./scripts/quick_train.sh
```

**See**: [`train-detection-local/README.md`](train-detection-local/README.md) for detailed training instructions.

### 2. Azure Deployment

Deploy the cloud infrastructure and functions:

```bash
# Navigate to Azure Functions
cd azure-function

# Install dependencies
npm install

# Deploy to Azure (requires Azure CLI)
./deploy-to-azure.sh
```

### 3. Integration

Connect your trained model to the Azure system:

1. Upload trained model (`train_detection_model.h5`) to Azure Blob Storage
2. Update Azure Function configuration to use your model
3. Test end-to-end pipeline with sample images

## 🎛 Components

### Azure Functions (Cloud Processing)

**Purpose**: Real-time monitoring and processing of uploaded images

**Key Features**:
- Automatic blob storage monitoring
- Image preprocessing and optimization
- Integration with AI detection models
- Result logging and notification
- Scalable serverless architecture

**Technology Stack**:
- TypeScript/Node.js
- Azure Functions Runtime
- Azure Storage SDK
- Azure AI Vision (optional fallback)

### Local ML Training (Model Development)

**Purpose**: Build and optimize custom train detection models

**Key Features**:
- Interactive dataset preparation
- CNN model training with TensorFlow
- Performance validation and testing
- Model export for cloud deployment
- Comprehensive training automation

**Technology Stack**:
- Python 3.8+
- TensorFlow/Keras
- OpenCV for image processing
- scikit-learn for metrics
- Matplotlib for visualization

## 📊 Performance Expectations

### Model Accuracy
- **Target**: >90% accuracy on validation set
- **Training Time**: 15-30 minutes (200 images per category)
- **Model Size**: 50-100MB
- **Inference Speed**: <1 second per image

### System Throughput
- **Image Processing**: 10+ images/minute
- **Azure Functions**: Auto-scaling based on load
- **Storage**: Unlimited via Azure Blob Storage
- **Latency**: 2-5 seconds end-to-end

## 🛠 Configuration

### Environment Variables

**Azure Functions**:
```bash
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection
AZURE_AI_VISION_ENDPOINT=your_vision_endpoint
AZURE_AI_VISION_KEY=your_vision_key
MODEL_BLOB_URL=your_trained_model_url
```

**Local Training**:
```bash
DATASET_PATH=./dataset
MODEL_OUTPUT_PATH=./models
TRAINING_EPOCHS=50
BATCH_SIZE=32
```

### Customization Options

**Image Processing**:
- Input image resolution
- Preprocessing filters
- Batch processing size

**Model Training**:
- CNN architecture depth
- Data augmentation settings
- Training hyperparameters

**Azure Integration**:
- Storage container names
- Function trigger sensitivity
- Notification settings

## 🧪 Testing

### Local Model Testing

```bash
cd train-detection-local

# Test individual images
python src/test_model.py

# Run comprehensive validation
./scripts/quick_test.sh
```

### Azure Function Testing

```bash
cd azure-function

# Run local function host
npm run start

# Upload test images to trigger processing
# Monitor function logs for results
```

### End-to-End Testing

1. Upload test images to Azure Blob Storage
2. Monitor Azure Function execution
3. Verify detection results in output logs
4. Validate accuracy against known ground truth

## 🔧 Troubleshooting

### Common Issues

**Model Training Problems**:
- Low accuracy: Need more/better training data
- Out of memory: Reduce batch size or image resolution
- Slow training: Enable GPU acceleration

**Azure Deployment Issues**:
- Function timeout: Optimize model inference speed
- Storage connection: Verify connection strings
- Permission errors: Check Azure role assignments

**Integration Problems**:
- Model loading errors: Verify model format compatibility
- Performance issues: Monitor resource utilization
- Data flow problems: Check blob trigger configuration

### Diagnostic Tools

**Local Development**:
- Training progress plots
- Validation metrics and confusion matrices
- Dataset validation reports

**Azure Monitoring**:
- Application Insights for function performance
- Storage Analytics for blob access patterns
- Function execution logs and metrics

## 📈 Scaling and Optimization

### Performance Optimization

**Model Optimization**:
- TensorFlow Lite conversion for faster inference
- Model quantization to reduce size
- Batch processing for multiple images

**Azure Optimization**:
- Premium Functions plan for consistent performance
- Container-based deployment for faster cold starts
- CDN integration for model distribution

### Scaling Strategies

**Horizontal Scaling**:
- Multiple Azure Function instances
- Distributed training across multiple machines
- Load-balanced inference endpoints

**Vertical Scaling**:
- Higher-tier Azure Function plans
- GPU-enabled training and inference
- Larger storage and compute resources

## 🔒 Security and Compliance

### Data Protection
- Encrypted storage for images and models
- Secure API endpoints with authentication
- GDPR-compliant data handling procedures

### Access Control
- Azure AD integration for user authentication
- Role-based access control (RBAC)
- API key management for external integrations

### Monitoring and Auditing
- Comprehensive logging of all processing activities
- Real-time alerting for system anomalies
- Regular security assessments and updates

## 📚 Documentation

- **[Local Training Guide](train-detection-local/README.md)**: Comprehensive model training documentation
- **[Azure Function Guide](azure-function/README.md)**: Cloud infrastructure setup
- **[Infrastructure Guide](azure-function/infrastructure/README.md)**: Bicep deployment documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to detect trains? Start with local model training in the [`train-detection-local`](train-detection-local/) directory! 🚂**
