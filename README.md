# LDSC-UNet-ViT

Lung Disease Segmentation and Classification using UNet and Vision Transformer

A complete full-stack medical image analysis system combining:
- **Frontend**: Modern React + TypeScript application with Material-UI
- **Backend**: Production-ready FastAPI with PyTorch deep learning models

## Project Overview

LDSC-UNet-ViT is a medical image analysis system for chest X-ray analysis combining:

1. **UNet** - For precise lung segmentation
2. **Vision Transformer (ViT)** - For 4-class disease classification (Normal, Pneumonia, COVID-19, Tuberculosis)

Features:
- 6-stage image preprocessing pipeline
- Real-time inference with Grad-CAM visualization
- Interactive React web interface
- RESTful API with auto-generated docs

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
npm install
npm run dev
```

API docs available at: http://localhost:8000/api/docs
