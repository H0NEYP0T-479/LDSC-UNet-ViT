from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
from app.models.unet_lung.infer_unet_lung import UNetInference
from app.utils.imaging import create_overlay
from app.services.storage_service import StorageService
from app.schemas.responses import SegmentationResult
from app.config import settings

router = APIRouter(prefix="/api", tags=["Segmentation"])
storage = StorageService(settings.artifacts_dir, settings.uploads_dir)


@router.post("/segment", response_model=SegmentationResult)
async def segment_image(file: UploadFile = File(...)):
    """Run UNet segmentation on uploaded X-ray."""
    try:
        contents = await file.read()
        image_path = storage.save_upload(contents, file.filename)

        unet = UNetInference(settings.checkpoint_unet, settings.device)
        unet.load_model()
        result = unet.predict_from_path(image_path)

        mask = result["mask"]
        import cv2
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        overlay = create_overlay(original, mask)

        return SegmentationResult(
            mask_url=storage.save_artifact(
                (mask * 255).astype(np.uint8), "mask"
            ),
            overlay_url=storage.save_artifact(overlay, "overlay"),
            disease_detected=result["disease_detected"],
            area_percentage=result["area_percentage"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))