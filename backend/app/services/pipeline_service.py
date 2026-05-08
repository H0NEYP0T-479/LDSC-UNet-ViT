import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np

from app.config import Settings
from app.models.vit.infer_vit import ViTInference
from app.models.unet_lung.infer_unet_lung import UNetInference
from app.utils.preprocessing import LDSCPreprocessor
from app.utils.gradcam import GradCAM
from app.utils.imaging import numpy_to_base64, create_overlay, save_image
from app.schemas.responses import (
    PreprocessingStages, SegmentationResult, ClassificationResult, InferenceResponse
)
from app.logging_config import get_logger


logger = get_logger(__name__)
_pipeline_instance = None


class PipelineService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vit_model = None
        self.unet_model = None
        self.preprocessor = LDSCPreprocessor()
        self.gradcam = None
        self.artifacts_dir = Path(settings.ARTIFACTS_DIR)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models(self) -> None:
        try:
            self.vit_model = ViTInference(
                checkpoint_path=str(self.settings.VIT_CHECKPOINT_PATH),
                device=str(self.settings.DEVICE)
            )
            logger.info(f"Loaded ViT model from {self.settings.VIT_CHECKPOINT_PATH}")
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            raise
        
        try:
            self.unet_model = UNetInference(
                checkpoint_path=str(self.settings.UNET_CHECKPOINT_PATH),
                device=str(self.settings.DEVICE)
            )
            logger.info(f"Loaded UNet model from {self.settings.UNET_CHECKPOINT_PATH}")
        except Exception as e:
            logger.error(f"Failed to load UNet model: {e}")
            raise
        
        self.gradcam = GradCAM(self.vit_model.model, self.vit_model.model.get_last_attention_layer())
    
    def run_inference(self, image_path: str, image_id: str = None) -> InferenceResponse:
        start_time = time.time()
        
        if image_id is None:
            image_id = str(uuid.uuid4())[:8]
        
        session_dir = self.artifacts_dir / image_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting inference pipeline for {image_id}")
        
        stages = self.preprocessor.get_stages(image_path)
        
        preprocessing_urls = self._save_preprocessing_stages(stages, session_dir)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            seg_future = executor.submit(self._run_segmentation, image_path, session_dir)
            clf_future = executor.submit(self._run_classification, image_path, session_dir)
            
            seg_result = seg_future.result()
            clf_result = clf_future.result()
        
        processing_time = time.time() - start_time
        
        response = InferenceResponse(
            preprocessing=preprocessing_urls,
            segmentation=seg_result,
            classification=clf_result,
            processing_time=processing_time,
            image_id=image_id
        )
        
        logger.info(f"Completed inference for {image_id} in {processing_time:.2f}s")
        
        return response
    
    def _save_preprocessing_stages(self, stages: dict, session_dir: Path) -> PreprocessingStages:
        urls = {}
        
        for stage_name, image_array in stages.items():
            if image_array is not None:
                image_path = session_dir / f"{stage_name}.png"
                save_image(str(image_path), image_array)
                urls[f"{stage_name}_url"] = numpy_to_base64(image_array)
        
        return PreprocessingStages(**urls)
    
    def _run_segmentation(self, image_path: str, session_dir: Path) -> SegmentationResult:
        seg_result = self.unet_model.predict_from_path(image_path)
        
        mask_path = session_dir / "segmentation_mask.png"
        save_image(str(mask_path), seg_result['mask'])
        mask_url = numpy_to_base64(seg_result['mask'])
        
        original_image = self.preprocessor.load_image(image_path)
        overlay_image = create_overlay(original_image, seg_result['mask'], alpha=0.5)
        
        overlay_path = session_dir / "segmentation_overlay.png"
        save_image(str(overlay_path), overlay_image)
        overlay_url = numpy_to_base64(overlay_image)
        
        return SegmentationResult(
            mask_url=mask_url,
            overlay_url=overlay_url,
            disease_detected=seg_result['tumor_detected'],
            area_percentage=seg_result['area_percentage']
        )
    
    def _run_classification(self, image_path: str, session_dir: Path) -> ClassificationResult:
        clf_result = self.vit_model.predict_from_path(image_path)
        
        original_image = self.preprocessor.load_image(image_path)
        
        try:
            vit_input = self.preprocessor.preprocess_for_vit(image_path)
            predicted_idx = list(clf_result['probabilities'].values()).index(
                max(clf_result['probabilities'].values())
            )
            
            heatmap = self.gradcam.generate(vit_input, class_idx=predicted_idx)
            heatmap_overlay = self.gradcam.overlay_heatmap(original_image, heatmap)
            
            gradcam_path = session_dir / "gradcam_heatmap.png"
            save_image(str(gradcam_path), heatmap_overlay)
        except Exception as e:
            logger.warning(f"GradCAM generation failed: {e}")
        
        return ClassificationResult(
            predicted_class=clf_result['predicted_class'],
            confidence=clf_result['confidence'],
            probabilities=clf_result['probabilities']
        )


def get_instance(settings: Settings = None) -> PipelineService:
    global _pipeline_instance
    
    if _pipeline_instance is None:
        if settings is None:
            settings = Settings()
        _pipeline_instance = PipelineService(settings)
        _pipeline_instance.load_models()
    
    return _pipeline_instance
