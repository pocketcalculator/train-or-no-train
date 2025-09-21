# Problem & Solution Overview

This is a solution to address automobile and pedestrian traffic problems created by stopped/parked trains.  This AI-powered train detection system combines local machine learning and edge device image detection with Azure cloud services for real-time monitoring and alerting of intersections impeded by stopped freight trains.

## 🎯 Technical Overview

This project provides an end-to-end solution for detecting trains in railroad track images using:

- **Local ML Training**: Build custom TensorFlow models on your Ubuntu machine
- **Azure Cloud Processing**: Scalable serverless image processing with Azure Functions
- **Real-time Monitoring**: Automated detection pipeline from image upload to results

## 🏗 Architecture

```
Local Training (Ubuntu)     →     Azure Cloud (Scalable)
┌─────────────────────┐           ┌─────────────────────┐
│  🤖 local-training  │    ───▶   │ ☁️ azure-function   │
│  • Dataset prep     │           │ • Blob monitoring   │
│  • Model training   │           │ • Image processing  │
│  • Validation       │           │ • Detection results │
│  • Export models    │           │ • Scalable inference│
└─────────────────────┘           └─────────────────────┘
```

## 📁 Project Structure

```
train-or-no-train/
├── 🤖 local-training/          # Local ML development (Ubuntu)
│   ├── src/                    # Python training scripts
│   ├── scripts/                # Automation scripts
│   ├── dataset/                # Training images
│   ├── models/                 # Trained models output
│   └── README.md               # Training guide
├── ☁️ azure-function/          # Cloud processing
│   ├── src/                    # Azure Function code
│   ├── infrastructure/         # Bicep templates
│   ├── test-scripts/           # Testing utilities
│   └── README.md               # Deployment guide
├── 🔗 shared/                  # Common code
│   ├── models/                 # Shared model definitions
│   ├── utils/                  # Common utilities
│   └── types/                  # Type definitions
├── 🧪 tests/                   # Testing
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
└── 📚 docs/                    # Documentation
    ├── ARCHITECTURE.md         # System architecture
    ├── PROJECT-STATUS.md       # Current status
    └── PROJECT-OVERVIEW.md     # Project details
```

## 🚀 Quick Start

### 1. Local Model Training (Start Here!)

Train your custom train detection model:

```bash
# Navigate to local training
cd local-training

# Set up environment
./setup.sh

# Prepare dataset
python src/dataset_helper.py

# Train model
./scripts/quick_train.sh
```

**📖 See**: [`local-training/README.md`](local-training/README.md) for detailed instructions.

### 2. Azure Deployment

Deploy to cloud after local training:

```bash
# Navigate to Azure components
cd azure-function

# Deploy infrastructure and function
./deploy-to-azure.sh
```

**📖 See**: [`azure-function/README.md`](azure-function/README.md) for deployment details.

## 🎛 Components

| Component | Purpose | Technology | Location |
|-----------|---------|------------|----------|
| **Local Training** | Build & train models | Python, TensorFlow | `local-training/` |
| **Azure Function** | Cloud processing | Node.js, TypeScript | `azure-function/` |
| **Infrastructure** | Cloud resources | Bicep, ARM templates | `azure-function/infrastructure/` |
| **Shared Code** | Common utilities | Python, TypeScript | `shared/` |
| **Documentation** | Guides & specs | Markdown | `docs/` |
| **Tests** | Quality assurance | Various | `tests/` |

## 📊 Performance Targets

- **Model Accuracy**: >90% on validation set
- **Training Time**: 15-30 minutes (200 images/category)
- **Inference Speed**: <1 second per image
- **Throughput**: 10+ images/minute in Azure
- **Model Size**: 50-100MB

## 🛠 Development Workflow

### Phase 1: Local Development
1. **Setup Environment** → `local-training/setup.sh`
2. **Prepare Dataset** → Collect and organize train/no-train images
3. **Train Model** → `local-training/scripts/quick_train.sh`
4. **Validate Performance** → `local-training/scripts/quick_test.sh`

### Phase 2: Cloud Integration
1. **Deploy Infrastructure** → `azure-function/deploy-to-azure.sh`
2. **Upload Trained Model** → Azure Blob Storage
3. **Configure Function** → Update environment variables
4. **Test End-to-End** → Upload test images

### Phase 3: Production
1. **Monitor Performance** → Azure Application Insights
2. **Scale Resources** → Based on load requirements
3. **Update Models** → Retrain and redeploy as needed

## 🧪 Testing

### Local Testing
```bash
cd local-training
./scripts/quick_test.sh
```

### Azure Testing
```bash
cd azure-function
./test-scripts/comprehensive-test.sh
```

### Integration Testing
```bash
cd tests
# Run integration test suites
```

## 📚 Documentation

- **[🏗 Architecture](docs/ARCHITECTURE.md)**: Complete system design
- **[📋 Project Status](docs/PROJECT-STATUS.md)**: What's ready and next steps
- **[🎯 Project Overview](docs/PROJECT-OVERVIEW.md)**: Detailed project information
- **[🤖 Local Training Guide](local-training/README.md)**: ML training instructions
- **[☁️ Azure Deployment Guide](azure-function/README.md)**: Cloud setup guide

## 🔧 Prerequisites

### Local Development
- **Ubuntu 24** (or compatible Linux)
- **Python 3.8+**
- **Git**

### Azure Deployment
- **Azure CLI**
- **Node.js 20+**
- **Azure Functions Core Tools v4**
- **Azure Subscription**

## 🎯 Current Status

✅ **Repository restructured** to optimal organization  
✅ **Local training environment** ready for use  
✅ **Azure infrastructure** prepared for deployment  
✅ **Documentation** comprehensive and up-to-date  
✅ **Testing frameworks** in place  

**🚀 Ready to begin**: Start with `cd local-training && ./setup.sh`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🚂 Ready to detect trains? Start with local training:** `cd local-training && ./setup.sh`