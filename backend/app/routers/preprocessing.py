from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from pathlib import Path

from app.config import Settings
from app.preprocessing import LDSCPreprocessor
from app.services.storage_service import StorageService
from app.logging_config import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix='/api')


@router.post('/preprocess')
async def preprocess_image(file: UploadFile = File(...)):
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
        
        logger.info(f"Processing uploaded image: {file.filename}")
        
        preprocessor = LDSCPreprocessor()
        stages = preprocessor.get_stages(image)
        
        stage_urls = {}
        for stage_name, stage_image in stages.items():
            try:
                url = storage_service.save_artifact(stage_image, f"preprocessing_{stage_name}")
                stage_urls[f"{stage_name}_url"] = url
            except Exception as e:
                logger.error(f"Failed to save {stage_name} stage: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to save {stage_name} stage")
        
        logger.info(f"Successfully preprocessed image: {file.filename}")
        
        return stage_urls
    
    except HTTPException as e:
        logger.error(f"HTTP Exception in preprocessing: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during preprocessing")
