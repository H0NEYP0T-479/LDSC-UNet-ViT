# LDSC-UNet-ViT Backend

Backend API for Lung Disease Segmentation and Classification using UNet and Vision Transformer.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
uvicorn app.main:app --reload
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Structure

- `app/main.py` - FastAPI application entry point
- `app/routers/` - API route handlers
- `app/models/` - ML models (UNet, ViT)
- `app/services/` - Business logic services
- `app/utils/` - Utility functions
- `app/schemas/` - Pydantic request/response schemas
- `app/resources/` - Model checkpoints and examples
