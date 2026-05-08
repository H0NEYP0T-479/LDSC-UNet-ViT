from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from pathlib import Path

from app.config import Settings
from app.preprocessing import LDSCPreprocessor
from app.models.vit.infer_vit import ViTInference
from app.services.pipeline_service import PipelineService
from app.services.storage_service import StorageService
from app.schemas.responses import ClassificationResult, InferenceResponse
from app.logging_config import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix='/api')


@router.post('/classify', response_model=ClassificationResult)
async def classify_disease(file: UploadFile = File(...)):
    try:
        settings = Settings()
        
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(status_code=422, detail="Failed to decode image")
        
        logger.info(f"Classifying image: {file.filename}")
        
        preprocessor = LDSCPreprocessor()
        stages = preprocessor.get_stages(image)
        normalized_image = stages['normalized']
        
        save_path = Path(settings.UPLOADS_DIR) / f"temp_{file.filename}"
        cv2.imwrite(str(save_path), normalized_image)
        
        try:
            device = 'cuda' if settings.USE_CUDA else 'cpu'
            vit_inference = ViTInference(settings.VIT_CHECKPOINT_PATH, device=device)
            
            result = vit_inference.predict_from_path(str(save_path))
            
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            probabilities = result['probabilities']
            
            logger.info(f"Classification complete for {file.filename}: class={predicted_class}, confidence={confidence:.4f}")
            
            return ClassificationResult(
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities
            )
        
        finally:
            if save_path.exists():
                save_path.unlink()
    
    except HTTPException as e:
        logger.error(f"HTTP Exception in classification: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in classification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")


@router.post('/inference', response_model=InferenceResponse)
async def run_full_inference(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(status_code=422, detail="Failed to decode image")
        
        settings = Settings()
        storage_path = Path(settings.UPLOADS_DIR) / f"temp_{file.filename}"
        cv2.imwrite(str(storage_path), image)
        
        try:
            logger.info(f"Starting full inference pipeline for: {file.filename}")
            
            pipeline_service = PipelineService.get_instance(settings)
            response = pipeline_service.run_inference(str(storage_path))
            
            logger.info(f"Full inference complete for {file.filename}: image_id={response.image_id}")
            
            return response
        
        finally:
            if storage_path.exists():
                storage_path.unlink()
    
    except HTTPException as e:
        logger.error(f"HTTP Exception in full inference: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in full inference: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during inference")
