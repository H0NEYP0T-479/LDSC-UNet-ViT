from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path
import torch
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "LDSC-UNet-ViT"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "Lung Disease Segmentation and Classification using UNet and Vision Transformer"
    DEBUG: bool = False

    # ==================== API Configuration ====================
    API_V1_STR: str = "/api"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # ==================== Paths Configuration ====================
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    BACKEND_DIR: Path = BASE_DIR / "backend"
    RESOURCES_DIR: Path = BACKEND_DIR / "app" / "resources"
    CHECKPOINTS_DIR: Path = RESOURCES_DIR / "checkpoints"
    EXAMPLES_DIR: Path = RESOURCES_DIR / "examples"

    # Data paths
    DATASET_ROOT: str = str(BACKEND_DIR / "data")
    UPLOAD_DIR: str = str(BACKEND_DIR / "uploads")
    ARTIFACTS_DIR: str = str(BACKEND_DIR / "artifacts")

    # ==================== Model Configuration ====================
    # ViT Model
    VIT_CHECKPOINT_PATH: str = str(CHECKPOINTS_DIR / "vit" / "vit_best.pth")
    VIT_IMAGE_SIZE: int = 224
    VIT_PATCH_SIZE: int = 16
    VIT_EMBED_DIM: int = 768
    VIT_NUM_HEADS: int = 12
    VIT_DEPTH: int = 12
    VIT_HIDDEN_DROPOUT_PROB: float = 0.1

    # UNet Model
    UNET_CHECKPOINT_PATH: str = str(CHECKPOINTS_DIR / "unet_lung" / "unet_lung_best.pth")
    UNET_IMAGE_SIZE: int = 256
    UNET_IN_CHANNELS: int = 1  # Grayscale medical images
    UNET_OUT_CHANNELS: int = 1  # Binary segmentation (lung vs background)
    UNET_DEPTH: int = 5

    # ==================== Disease Classification ====================
    CLASS_NAMES: List[str] = ["normal", "pneumonia", "covid19", "tuberculosis"]
    NUM_CLASSES: int = 4

    # ==================== Image Processing ====================
    IMAGE_SIZE_VIT: int = 224
    IMAGE_SIZE_UNET: int = 256
    IMAGE_MEAN: List[float] = [0.485]  # For grayscale
    IMAGE_STD: List[float] = [0.229]   # For grayscale

    # ==================== Training Configuration ====================
    BATCH_SIZE: int = 32
    VAL_BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_EPOCHS: int = 100
    WARMUP_EPOCHS: int = 10

    # ==================== Device Configuration ====================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4 if torch.cuda.is_available() else 0
    PIN_MEMORY: bool = torch.cuda.is_available()

    # ==================== Logging Configuration ====================
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = str(BACKEND_DIR / "logs")
    LOG_FILE: str = "app.log"

    # ==================== Upload Configuration ====================
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "dcm", "dicom"]
    ALLOWED_MIMETYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "application/dicom",
    ]

    # ==================== Inference Configuration ====================
    INFERENCE_CONFIDENCE_THRESHOLD: float = 0.5
    INFERENCE_BATCH_SIZE: int = 1

    # ==================== Pydantic Configuration ====================
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra environment variables


def get_settings() -> Settings:
    return settings


# Create a single settings instance
settings = Settings()

# Create necessary directories if they don't exist
os.makedirs(settings.DATASET_ROOT, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)
os.makedirs(settings.CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(settings.EXAMPLES_DIR, exist_ok=True)
