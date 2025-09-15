# Azure Function - Train Detection Integration

This Azure Function component provides the cloud infrastructure for real-time train detection processing. It integrates with your locally-trained TensorFlow model to provide scalable, serverless image processing.

## 🎯 Purpose

The Azure Function serves as the cloud orchestration layer that:
- Monitors blob storage for new railroad track images
- Downloads and preprocesses uploaded images
- Runs train detection inference using your custom model
- Logs detection results and manages processed files
- Provides scalable, serverless processing architecture

## 🔄 Integration with Local Training

This Azure Function is designed to work seamlessly with the models you train locally in the `train-detection-local` directory:

1. **Train your model locally** using the Python scripts in `../train-detection-local/`
2. **Upload your trained model** (`train_detection_model.h5`) to Azure Blob Storage
3. **Configure this function** to use your uploaded model
4. **Deploy and run** the function to process images in real-time

## 📁 Current Structure

```
azure-function/
├── src/functions/
│   ├── blobMonitor.ts              # Original blob monitoring function
│   └── enhancedBlobMonitor.ts      # Enhanced function with train detection
├── infrastructure/
│   ├── main.bicep                  # Azure infrastructure as code
│   └── main.bicepparam             # Deployment parameters
├── test-scripts/
│   ├── test-helper.sh              # Development utilities
│   ├── upload-test-files.sh        # Test image uploads
│   ├── monitor-logs.sh             # Real-time log monitoring
│   └── comprehensive-test.sh       # Full system testing
├── package.json                    # Node.js dependencies
├── tsconfig.json                   # TypeScript configuration
├── host.json                       # Function runtime configuration
└── local.settings.json             # Local development settings
```

## 🚀 Quick Setup

### Prerequisites
- Node.js 20+ 
- Azure Functions Core Tools v4
- Azure CLI (for deployment)
- Azure Storage Account

### Local Development
```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Start local development
npm run start
```

### Azure Deployment
```bash
# Deploy infrastructure and function
./deploy-to-azure.sh
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_STORAGE_CONNECTION_STRING` | Storage account connection string | `DefaultEndpointsProtocol=https;AccountName=...` |
| `INCOMING_CONTAINER_NAME` | Container for new images | `incoming-images` |
| `PROCESSED_CONTAINER_NAME` | Container for processed images | `processed-images` |
| `MODEL_BLOB_URL` | URL to your trained model file | `https://storage.blob.core.windows.net/.../model.h5` |
| `AZURE_AI_VISION_ENDPOINT` | Optional: Azure AI Vision endpoint | `https://region.cognitiveservices.azure.com/` |
| `AZURE_AI_VISION_KEY` | Optional: Azure AI Vision API key | `your-api-key` |

### Function Settings

The enhanced blob monitor function (`enhancedBlobMonitor.ts`) includes:
- **Timer Trigger**: Configurable interval (default: every 30 seconds)
- **Image Processing**: Automatic download and preprocessing
- **Model Integration**: Loads your TensorFlow model for inference
- **Result Logging**: Comprehensive detection results and metrics
- **Error Handling**: Robust error handling with detailed logging

## 🔄 Workflow

### 1. Image Upload
Images are uploaded to the `incoming-images` container via:
- Automated camera systems
- Manual upload via Azure Portal
- Programmatic upload via SDK/CLI

### 2. Function Trigger
The Azure Function automatically triggers when:
- Timer interval elapses (every 30 seconds by default)
- New images are detected in the incoming container

### 3. Image Processing
For each detected image:
1. Download image from blob storage
2. Preprocess image (resize, normalize)
3. Load your trained TensorFlow model
4. Run inference to detect trains
5. Log detection results with confidence scores

### 4. Result Management
After processing:
- Move processed images to `processed-images` container
- Log results with timestamps and confidence scores
- Optional: Send notifications or webhooks for positive detections

## 🧪 Testing

### Local Testing with Azurite
```bash
# Start Azurite emulator
azurite --silent --location ./azurite-data

# Create test images
./test-scripts/test-helper.sh test-files

# Upload test images
./test-scripts/upload-test-files.sh

# Monitor function execution
./test-scripts/monitor-logs.sh
```

### End-to-End Testing
```bash
# Run comprehensive test suite
./test-scripts/comprehensive-test.sh
```

This will:
1. Deploy test infrastructure
2. Upload sample images (with and without trains)
3. Monitor function execution
4. Validate detection results
5. Generate performance report

## 📊 Monitoring and Logging

### Function Logs
The function provides detailed logging including:
- Image processing timestamps
- Model loading and inference times
- Detection results with confidence scores
- Error messages and stack traces
- Performance metrics

### Azure Monitoring
Integration with Azure services:
- **Application Insights**: Performance monitoring and alerting
- **Storage Analytics**: Blob access patterns and performance
- **Function Metrics**: Execution count, duration, and success rates

### Log Examples
```
[2024-01-15 10:30:45] Processing image: train_image_001.png
[2024-01-15 10:30:46] Model loaded successfully: train_detection_model.h5
[2024-01-15 10:30:47] Detection result: TRAIN DETECTED (confidence: 0.94)
[2024-01-15 10:30:47] Moved to processed container: train_image_001.png
```

## 🔄 Model Integration

### Using Your Locally Trained Model

1. **Train your model** using the scripts in `../train-detection-local/`
2. **Upload the model file** to Azure Blob Storage:
   ```bash
   az storage blob upload \
     --account-name your-storage-account \
     --container-name models \
     --name train_detection_model.h5 \
     --file ../train-detection-local/models/train_detection_model.h5
   ```

3. **Update configuration** with your model URL:
   ```bash
   # Set environment variable
   export MODEL_BLOB_URL="https://yourstorage.blob.core.windows.net/models/train_detection_model.h5"
   ```

4. **Redeploy the function** to use your new model

### Model Performance
The function tracks and logs:
- Model loading time
- Inference time per image
- Memory usage during processing
- Detection accuracy (when ground truth is available)

## 🔧 Troubleshooting

### Common Issues

**Function Not Triggering**:
- Check timer trigger configuration
- Verify storage container exists and has images
- Review function logs for errors

**Model Loading Errors**:
- Verify model file URL is accessible
- Check model file format (should be .h5)
- Ensure sufficient memory allocation

**Poor Detection Performance**:
- Validate image preprocessing matches training data
- Check model confidence thresholds
- Verify image quality and format

**Storage Connection Issues**:
- Verify storage connection string
- Check container names and permissions
- Test blob access with Azure CLI

### Performance Optimization

**For High Volume Processing**:
- Increase function timeout settings
- Use Premium Functions plan for consistent performance
- Consider batch processing multiple images

**For Low Latency**:
- Keep function "warm" with regular pings
- Optimize model size and inference speed
- Use container-based deployment

## 🚀 Deployment Options

### Development Deployment
```bash
# Deploy to development environment
func azure functionapp publish your-dev-function-app --slot staging
```

### Production Deployment
```bash
# Deploy infrastructure first
az deployment group create \
  --resource-group your-rg \
  --template-file infrastructure/main.bicep \
  --parameters infrastructure/main.bicepparam

# Deploy function code
func azure functionapp publish your-prod-function-app
```

### CI/CD Integration
The project includes GitHub Actions workflow templates for:
- Automated testing on pull requests
- Infrastructure deployment to Azure
- Function app deployment with validation
- Model update notifications

## 📈 Scaling Considerations

### Traffic Patterns
- **Low Traffic**: Consumption plan with timer triggers
- **Medium Traffic**: Premium plan with blob triggers
- **High Traffic**: Dedicated App Service plan with multiple instances

### Performance Metrics
Monitor these key metrics:
- Function execution duration
- Memory consumption during model inference
- Storage I/O performance
- Detection accuracy and false positive rates

### Cost Optimization
- Use consumption plan for variable workloads
- Optimize image sizes and formats
- Implement intelligent caching for frequently accessed models
- Monitor and alert on cost thresholds

## 📚 Next Steps

1. **Complete Local Training**: Finish training your model in `../train-detection-local/`
2. **Upload Your Model**: Deploy your trained model to Azure Blob Storage
3. **Configure Function**: Update environment variables with your model URL
4. **Test Integration**: Run end-to-end tests with your trained model
5. **Deploy to Production**: Use the deployment scripts for production deployment

## 🔗 Related Documentation

- **[Local Training Guide](../train-detection-local/README.md)**: Complete model training documentation
- **[Main Project README](../README.md)**: Overall system architecture and setup
- **[Infrastructure Guide](infrastructure/README.md)**: Azure resource deployment details

---

**Ready to integrate your trained model with Azure? Follow the model integration steps above! ☁️**