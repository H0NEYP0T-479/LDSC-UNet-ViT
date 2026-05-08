from pydantic import BaseModel
from typing import Dict


class PreprocessingStages(BaseModel):
    """URLs for each preprocessing stage image."""
    original_url: str
    grayscale_url: str
    denoised_url: str
    enhanced_url: str
    sharpened_url: str
    normalized_url: str


class SegmentationResult(BaseModel):
    """UNet segmentation result."""
    mask_url: str
    overlay_url: str
    disease_detected: bool
    area_percentage: float


class ClassificationResult(BaseModel):
    """ViT classification result."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class InferenceResponse(BaseModel):
    """Full pipeline inference response."""
    image_id: str
    preprocessing: PreprocessingStages
    segmentation: SegmentationResult
    classification: ClassificationResult
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models: Dict[str, str]
    version: str