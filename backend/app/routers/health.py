from fastapi import APIRouter
from pathlib import Path

from app.config import Settings
from app.schemas.responses import HealthResponse
from app.logging_config import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix='/api')


@router.get('/health', response_model=HealthResponse)
def health_check():
    settings = Settings()
    
    vit_loaded = Path(settings.VIT_CHECKPOINT_PATH).exists()
    unet_loaded = Path(settings.UNET_CHECKPOINT_PATH).exists()
    
    models = {
        'vit': 'loaded' if vit_loaded else 'not_loaded',
        'unet': 'loaded' if unet_loaded else 'not_loaded'
    }
    
    status = 'healthy' if (vit_loaded and unet_loaded) else 'degraded'
    
    logger.info(f"Health check: status={status}, models={models}")
    
    return HealthResponse(
        status=status,
        models=models,
        version='1.0.0'
    )
