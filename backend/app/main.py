from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import Settings
from app.logging_config import setup_logging, get_logger
from app.routers import health, preprocessing, segmentation, classification
from app.services.pipeline_service import PipelineService


logger = get_logger(__name__)
settings = Settings()

app = FastAPI(
    title="LDSC-UNet-ViT API",
    description="Lung Disease Segmentation and Classification using UNet and Vision Transformer",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

artifacts_path = Path(settings.ARTIFACTS_DIR)
artifacts_path.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(artifacts_path)), name="artifacts")

app.include_router(health.router, tags=["health"])
app.include_router(preprocessing.router, tags=["preprocessing"])
app.include_router(segmentation.router, tags=["segmentation"])
app.include_router(classification.router, tags=["classification"])


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    
    Initializes the application by:
    - Setting up structured logging with configured log level
    - Loading ViT and UNet models via PipelineService singleton
    - Validating model checkpoints are available
    
    Raises:
        Exception: If logging setup or model loading fails
    """
    try:
        setup_logging(settings.LOG_LEVEL)
        logger.info("Logging configured successfully")
        
        pipeline_service = PipelineService.get_instance(settings)
        pipeline_service.load_models()
        logger.info("Models loaded successfully on startup")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    
    Performs cleanup on application shutdown:
    - Logs shutdown event
    - Releases model resources
    - Closes file handles
    """
    logger.info("Application shutting down")


@app.get("/")
async def root():
    """
    Root endpoint - returns welcome message and API information.
    
    Returns:
        dict: Welcome message with API version, description, and endpoint URLs
    """
    return {
        "message": "LDSC-UNet-ViT API",
        "description": "Lung Disease Segmentation and Classification using UNet and Vision Transformer",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "preprocessing": "/api/preprocess",
            "segmentation": "/api/segment",
            "classification": "/api/classify",
            "full_inference": "/api/inference",
            "docs": "/api/docs",
            "artifacts": "/artifacts"
        }
    }
