from pydantic import BaseModel
from typing import Dict, Optional


class PreprocessingStages(BaseModel):
    original_url: str
    grayscale_url: str
    denoised_url: str
    enhanced_url: str
    sharpened_url: str
    normalized_url: str


class SegmentationResult(BaseModel):
    mask_url: str
    overlay_url: str
    disease_detected: bool
    area_percentage: float


class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class InferenceResponse(BaseModel):
    preprocessing: PreprocessingStages
    segmentation: SegmentationResult
    classification: ClassificationResult
    processing_time: float
    image_id: str


class HealthResponse(BaseModel):
    status: str
    models: Dict[str, str]
    version: str
