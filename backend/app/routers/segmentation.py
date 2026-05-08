from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from pathlib import Path

from app.config import Settings
from app.preprocessing import LDSCPreprocessor
from app.imaging import create_overlay
from app.models.unet_lung.infer_unet_lung import UNetInference
from app.services.storage_service import StorageService
from app.schemas.responses import SegmentationResult
from app.logging_config import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix='/api')


@router.post('/segment', response_model=SegmentationResult)
async def segment_lung(file: UploadFile = File(...)):
    try:
        settings = Settings()
        storage_service = StorageService(settings.ARTIFACTS_DIR, settings.UPLOADS_DIR)
        
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(status_code=422, detail="Failed to decode image")
        
        logger.info(f"Segmenting image: {file.filename}")
        
        preprocessor = LDSCPreprocessor()
        stages = preprocessor.get_stages(image)
        normalized_image = stages['normalized']
        
        save_path = Path(settings.UPLOADS_DIR) / f"temp_{file.filename}"
        cv2.imwrite(str(save_path), normalized_image)
        
        try:
            device = 'cuda' if settings.USE_CUDA else 'cpu'
            unet_inference = UNetInference(settings.UNET_CHECKPOINT_PATH, device=device)
            
            seg_result = unet_inference.predict_from_path(str(save_path))
            
            mask = seg_result['mask']
            disease_detected = seg_result['tumor_detected']
            area_percentage = seg_result['area_percentage']
            
            mask_url = storage_service.save_artifact(mask, f"segmentation_mask")
            
            overlay = create_overlay(image, mask)
            overlay_url = storage_service.save_artifact(overlay, f"segmentation_overlay")
            
            logger.info(f"Segmentation complete for {file.filename}: detected={disease_detected}, area={area_percentage}%")
            
            return SegmentationResult(
                mask_url=mask_url,
                overlay_url=overlay_url,
                disease_detected=disease_detected,
                area_percentage=area_percentage
            )
        
        finally:
            if save_path.exists():
                save_path.unlink()
    
    except HTTPException as e:
        logger.error(f"HTTP Exception in segmentation: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in segmentation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during segmentation")
