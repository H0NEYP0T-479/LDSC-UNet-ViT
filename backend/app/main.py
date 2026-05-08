from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from app.config import settings
from app.logging_config import setup_logging
from app.routers import health, preprocessing, segmentation, classification
from app.services.pipeline_service import get_instance

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Lung Disease Segmentation and Classification API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs(settings.artifacts_dir, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=settings.artifacts_dir), name="artifacts")

app.include_router(health.router)
app.include_router(preprocessing.router)
app.include_router(segmentation.router)
app.include_router(classification.router)


@app.on_event("startup")
async def startup():
    setup_logging(settings.log_level)
    pipeline = get_instance(settings)
    try:
        pipeline.load_models()
    except Exception as e:
        print(f"Warning: Models not loaded - {e}")


@app.on_event("shutdown")
async def shutdown():
    print("Shutting down LDSC-UNet-ViT API")


@app.get("/")
async def root():
    return {
        "message": "Welcome to LDSC-UNet-ViT API",
        "version": settings.version,
        "docs": "/docs"
    }