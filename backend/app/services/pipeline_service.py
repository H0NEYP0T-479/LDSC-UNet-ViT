import os
import time
import uuid
import cv2
import numpy as np
from app.models.vit.infer_vit import ViTInference
from app.models.unet_lung.infer_unet_lung import UNetInference
from app.utils.preprocessing import LDSCPreprocessor
from app.utils.gradcam import GradCAM
from app.utils.imaging import create_overlay
from app.schemas.responses import (
    InferenceResponse, PreprocessingStages,
    SegmentationResult, ClassificationResult
)
from app.services.storage_service import StorageService

_instance = None


class PipelineService:
    """End-to-end inference pipeline service."""

    def __init__(self, settings):
        self.settings = settings
        self.vit = None
        self.unet = None
        self.preprocessor = LDSCPreprocessor()
        self.storage = StorageService(
            settings.artifacts_dir,
            settings.uploads_dir
        )

    def load_models(self) -> None:
        """Load ViT and UNet models from checkpoints."""
        self.vit = ViTInference(
            self.settings.checkpoint_vit,
            self.settings.device
        )
        self.vit.load_model()

        self.unet = UNetInference(
            self.settings.checkpoint_unet,
            self.settings.device
        )
        self.unet.load_model()

    def run_inference(self, image_path: str) -> InferenceResponse:
        """Run full pipeline on image path."""
        start = time.time()
        image_id = uuid.uuid4().hex

        # Step 1: Preprocessing stages
        stages = self.preprocessor.get_stages(image_path)
        preprocessing = PreprocessingStages(
            original_url=self.storage.save_artifact(
                stages["original"], "original"
            ),
            grayscale_url=self.storage.save_artifact(
                stages["grayscale"], "grayscale"
            ),
            denoised_url=self.storage.save_artifact(
                stages["denoised"], "denoised"
            ),
            enhanced_url=self.storage.save_artifact(
                stages["enhanced"], "enhanced"
            ),
            sharpened_url=self.storage.save_artifact(
                stages["sharpened"], "sharpened"
            ),
            normalized_url=self.storage.save_artifact(
                stages["normalized"], "normalized"
            )
        )

        # Step 2: ViT classification
        vit_result = self.vit.predict_from_path(image_path)
        classification = ClassificationResult(**vit_result)

        # Step 3: UNet segmentation
        unet_result = self.unet.predict_from_path(image_path)
        mask = unet_result["mask"]
        original_gray = stages["grayscale"]
        overlay = create_overlay(original_gray, mask)

        segmentation = SegmentationResult(
            mask_url=self.storage.save_artifact(
                (mask * 255).astype(np.uint8), "mask"
            ),
            overlay_url=self.storage.save_artifact(overlay, "overlay"),
            disease_detected=unet_result["disease_detected"],
            area_percentage=unet_result["area_percentage"]
        )

        return InferenceResponse(
            image_id=image_id,
            preprocessing=preprocessing,
            segmentation=segmentation,
            classification=classification,
            processing_time=round(time.time() - start, 3)
        )


def get_instance(settings=None) -> PipelineService:
    """Return singleton PipelineService instance."""
    global _instance
    if _instance is None and settings is not None:
        _instance = PipelineService(settings)
    return _instance