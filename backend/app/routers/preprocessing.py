from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.preprocessing import LDSCPreprocessor
from app.services.storage_service import StorageService
from app.config import settings

router = APIRouter(prefix="/api", tags=["Preprocessing"])
preprocessor = LDSCPreprocessor()
storage = StorageService(settings.artifacts_dir, settings.uploads_dir)


@router.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """Preprocess uploaded X-ray and return stage image URLs."""
    try:
        contents = await file.read()
        image_path = storage.save_upload(contents, file.filename)
        stages = preprocessor.get_stages(image_path)
        return {
            "original_url": storage.save_artifact(stages["original"], "original"),
            "grayscale_url": storage.save_artifact(stages["grayscale"], "grayscale"),
            "denoised_url": storage.save_artifact(stages["denoised"], "denoised"),
            "enhanced_url": storage.save_artifact(stages["enhanced"], "enhanced"),
            "sharpened_url": storage.save_artifact(stages["sharpened"], "sharpened"),
            "normalized_url": storage.save_artifact(stages["normalized"], "normalized")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))