# Tests

This directory contains all testing infrastructure for the train detection system.

## 📁 Structure

- **`unit/`**: Unit tests for individual components
- **`integration/`**: Integration tests between components
- **`e2e/`**: End-to-end tests for the complete system

## 🧪 Test Categories

### Unit Tests
- Local training script validation
- Azure Function logic testing
- Utility function verification
- Model validation tests

### Integration Tests
- Local-to-Azure model deployment
- Azure Function with storage integration
- End-to-end data pipeline validation

### End-to-End Tests
- Complete workflow from image upload to detection
- Performance and scalability testing
- Error handling and recovery testing

## 🚀 Running Tests

### Local Testing
```bash
# From repository root
cd tests/unit
python -m pytest

# Test specific components
cd tests/integration
python test_local_azure_integration.py
```

### Azure Testing
```bash
# Test Azure components
cd tests/e2e
./test_azure_pipeline.sh
```

---

**Test suites will be added as components are developed and integrated.**