# Shared Utilities

This directory contains code and resources shared between the local training environment and Azure cloud components.

## 📁 Structure

- **`models/`**: Shared model definitions and schemas
- **`utils/`**: Common utility functions
- **`types/`**: Type definitions for both Python and TypeScript

## 🔗 Usage

These shared components are used by:
- Local training scripts (`../local-training/`)
- Azure Functions (`../azure-function/`)
- Testing frameworks (`../tests/`)

## 📦 Components

### Models
- Model interface definitions
- Common model validation schemas
- Model conversion utilities

### Utils
- Image preprocessing functions
- Data validation utilities
- Configuration helpers

### Types
- Common data structures
- API interfaces
- Model input/output types

---

**This directory will be populated as the project develops shared functionality.**