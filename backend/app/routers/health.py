import os
from fastapi import APIRouter
from app.schemas.responses import HealthResponse
from app.config import settings

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model availability."""
    return HealthResponse(
        status="healthy",
        models={
            "vit": "loaded" if os.path.exists(settings.checkpoint_vit)
                   else "not_loaded",
            "unet": "loaded" if os.path.exists(settings.checkpoint_unet)
                    else "not_loaded"
        },
        version=settings.version
    )