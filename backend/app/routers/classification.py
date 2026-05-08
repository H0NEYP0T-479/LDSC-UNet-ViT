from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.vit.infer_vit import ViTInference
from app.services.pipeline_service import get_instance
from app.services.storage_service import StorageService
from app.schemas.responses import ClassificationResult, InferenceResponse
from app.config import settings

router = APIRouter(prefix="/api", tags=["Classification"])
storage = StorageService(settings.artifacts_dir, settings.uploads_dir)


@router.post("/classify", response_model=ClassificationResult)
async def classify_image(file: UploadFile = File(...)):
    """Run ViT classification on uploaded X-ray."""
    try:
        contents = await file.read()
        image_path = storage.save_upload(contents, file.filename)
        vit = ViTInference(settings.checkpoint_vit, settings.device)
        vit.load_model()
        result = vit.predict_from_path(image_path)
        return ClassificationResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference", response_model=InferenceResponse)
async def full_inference(file: UploadFile = File(...)):
    """Run full pipeline: preprocess + classify + segment."""
    try:
        contents = await file.read()
        image_path = storage.save_upload(contents, file.filename)
        pipeline = get_instance(settings)
        return pipeline.run_inference(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))