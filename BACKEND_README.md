# LDSC-UNet-ViT Backend

Production-ready FastAPI backend for Lung Disease Segmentation and Classification using UNet and Vision Transformer.

## Features

- **FastAPI** - Modern, fast web framework
- **Deep Learning Models**:
  - **UNet** - For precise lung segmentation
  - **Vision Transformer (ViT)** - For disease classification
- **Medical Image Processing** - 6-stage preprocessing pipeline
- **Grad-CAM** - Model interpretability visualization
- **Async Support** - Concurrent request handling
- **CORS Enabled** - Frontend integration ready
- **Structured Logging** - Complete request tracing
- **Static File Serving** - Artifact management

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration settings
│   ├── logging_config.py    # Logging setup
│   ├── routers/             # API endpoints
│   │   ├── health.py        # Health check
│   │   ├── preprocessing.py # Image preprocessing
│   │   ├── segmentation.py  # Lung segmentation
│   │   └── classification.py # Disease classification
│   ├── services/            # Business logic
│   │   ├── pipeline_service.py    # End-to-end pipeline
│   │   └── storage_service.py     # File management
│   ├── models/              # Deep learning models
│   │   ├── vit/             # Vision Transformer
│   │   │   ├── model.py
│   │   │   ├── datamodule.py
│   │   │   ├── train_vit.py
│   │   │   └── infer_vit.py
│   │   └── unet_lung/       # UNet Segmentation
│   │       ├── model.py
│   │       ├── datamodule.py
│   │       ├── train_unet_lung.py
│   │       └── infer_unet_lung.py
│   ├── schemas/             # Request/response schemas
│   │   ├── requests.py
│   │   └── responses.py
│   └── utils/               # Utility functions
│       ├── logger.py        # Logging utilities
│       ├── preprocessing.py # Image preprocessing
│       ├── imaging.py       # Image I/O operations
│       ├── gradcam.py       # Grad-CAM visualization
│       └── metrics.py       # Medical imaging metrics
├── resources/
│   └── checkpoints/         # Model checkpoints directory
├── logs/                    # Application logs
├── uploads/                 # Temporary uploads
├── artifacts/               # Processing results
└── requirements.txt         # Python dependencies
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/LDSC-UNet-ViT.git
cd LDSC-UNet-ViT/backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Configure settings in `app/config.py` or `.env`

### Run Application

**Development**:
```bash
uvicorn app.main:app --reload
```

**Production**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access API docs at: `http://localhost:8000/api/docs`

## API Endpoints

### Health Check
```
GET /api/health
```
Returns system and model status.

### Image Preprocessing
```
POST /api/preprocess
Content-Type: multipart/form-data
```
Request:
- `file`: Medical image (JPEG, PNG)

Response:
```json
{
  "original_url": "/artifacts/preprocessing_original_xxx.png",
  "grayscale_url": "/artifacts/preprocessing_grayscale_xxx.png",
  "denoised_url": "/artifacts/preprocessing_denoised_xxx.png",
  "enhanced_url": "/artifacts/preprocessing_enhanced_xxx.png",
  "sharpened_url": "/artifacts/preprocessing_sharpened_xxx.png",
  "normalized_url": "/artifacts/preprocessing_normalized_xxx.png"
}
```

### Lung Segmentation
```
POST /api/segment
Content-Type: multipart/form-data
```
Request:
- `file`: Medical image

Response:
```json
{
  "mask_url": "/artifacts/segmentation_mask_xxx.png",
  "overlay_url": "/artifacts/segmentation_overlay_xxx.png",
  "disease_detected": true,
  "area_percentage": 23.5
}
```

### Disease Classification
```
POST /api/classify
Content-Type: multipart/form-data
```
Request:
- `file`: Medical image

Response:
```json
{
  "predicted_class": "pneumonia",
  "confidence": 0.95,
  "probabilities": {
    "normal": 0.02,
    "pneumonia": 0.95,
    "covid19": 0.02,
    "tuberculosis": 0.01
  }
}
```

### Full Pipeline Inference
```
POST /api/inference
Content-Type: multipart/form-data
```
Request:
- `file`: Medical image

Response:
```json
{
  "preprocessing": { /* 6 preprocessing stages */ },
  "segmentation": { /* segmentation results */ },
  "classification": { /* classification results */ },
  "gradcam_url": "/artifacts/gradcam_xxx.png",
  "processing_time": 2.35,
  "image_id": "img_xxx"
}
```

## Configuration

### Environment Variables

```bash
# API
API_V1_STR=/api
DEBUG=False

# Device
DEVICE=cuda  # or cpu

# Paths
DATASET_ROOT=backend/data
UPLOAD_DIR=backend/uploads
ARTIFACTS_DIR=backend/artifacts

# Model Checkpoints
VIT_CHECKPOINT_PATH=backend/app/resources/checkpoints/vit/vit_best.pth
UNET_CHECKPOINT_PATH=backend/app/resources/checkpoints/unet_lung/unet_lung_best.pth

# Image Sizes
IMAGE_SIZE_VIT=224
IMAGE_SIZE_UNET=256

# Inference
INFERENCE_CONFIDENCE_THRESHOLD=0.5
```

## Models

### Vision Transformer (ViT)

- **Architecture**: ViT Base Patch16-224
- **Pre-training**: ImageNet-21k
- **Input**: 224x224 RGB
- **Output**: 4 disease classes
- **Classes**: normal, pneumonia, covid19, tuberculosis

### UNet Segmentation

- **Architecture**: 5-level encoder-decoder
- **Input**: 256x256 grayscale
- **Output**: 256x256 binary mask
- **Loss**: Combined Dice + BCE

## Performance

- **Inference Time**: ~2-3 seconds per image
- **Throughput**: ~20-30 images/minute (single GPU)
- **Memory Usage**: ~2GB GPU for concurrent requests
- **Accuracy**: 
  - ViT Classification: ~92% (varies by disease)
  - UNet Segmentation: ~0.85 Dice score

## Preprocessing Pipeline

1. **Original** - Input image
2. **Grayscale** - Convert to grayscale
3. **Denoise** - Non-local means denoising
4. **Enhanced** - Contrast enhancement (CLAHE)
5. **Sharpen** - Edge sharpening
6. **Normalized** - Normalization for model input

## Logging

- **Console**: INFO level
- **File**: `backend/logs/app.log`
- **Rotation**: 10MB with 5 backups
- **Format**: Timestamp, Level, Module, Message

Example log:
```
2024-01-15 10:30:45,123 - app.routers.segmentation - INFO - [image_001 | segmentation] Completed inference
```

## Error Handling

- HTTP status codes (400, 422, 500)
- Detailed error messages
- Full traceback logging
- Graceful fallback

## Development

### Training Models

```bash
# Train ViT
python -m app.models.vit.train_vit

# Train UNet
python -m app.models.unet_lung.train_unet_lung
```

### Testing

```bash
pytest backend/tests/
```

### Code Quality

- Type hints throughout
- Docstrings on functions
- Error handling
- Logging

## Performance Optimization

- Model caching with PipelineService singleton
- Parallel inference (ViT + UNet concurrent)
- GradScaler for mixed precision training
- Efficient image I/O with OpenCV

## Security

- CORS configured for frontend
- Input validation via Pydantic
- File size limits (50MB max)
- Supported formats whitelist

## Future Enhancements

- [ ] Batch processing endpoint
- [ ] Model ensemble
- [ ] Real-time progress streaming
- [ ] Database for result storage
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Model versioning
- [ ] A/B testing support
- [ ] Performance monitoring
- [ ] Automated model retraining

## Troubleshooting

### GPU Memory Error
- Reduce batch size in config
- Use CPU mode: `DEVICE=cpu`

### Model Loading Error
- Verify checkpoint paths
- Check model compatibility
- Run health endpoint

### Slow Performance
- Check GPU availability
- Monitor memory usage
- Review preprocessing step

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [UNet Paper](https://arxiv.org/abs/1505.04597)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)

## License

MIT License - See LICENSE file

## Contact

For issues, questions, or contributions, please reach out to the development team.
