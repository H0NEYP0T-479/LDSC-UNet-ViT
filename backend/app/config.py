from pydantic_settings import BaseSettings
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Project
    project_name: str = "LDSC-UNet-ViT"
    version: str = "1.0.0"
    debug: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:5173"]

    # Paths
    checkpoint_vit: str = str(BASE_DIR / "resources/checkpoints/vit/vit_best.pth")
    checkpoint_unet: str = str(BASE_DIR / "resources/checkpoints/unet_lung/unet_lung_best.pth")
    uploads_dir: str = str(BASE_DIR / "resources/uploads")
    artifacts_dir: str = str(BASE_DIR / "resources/artifacts")
    dataset_root: str = str(BASE_DIR / "resources/dataset")

    # Image sizes
    vit_image_size: int = 224
    unet_image_size: int = 256

    # Classes
    class_names: list[str] = ["normal", "pneumonia", "covid19", "tuberculosis"]

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()