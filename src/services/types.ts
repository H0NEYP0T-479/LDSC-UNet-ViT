export interface PreprocessingResult {
  status: string;
  image_id: string;
  technique_applied: string;
}

export interface SegmentationResult {
  status: string;
  image_id: string;
  mask_id: string;
  confidence: number;
}

export interface ClassificationResult {
  status: string;
  image_id: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
}

export interface ModelStatus {
  models: {
    unet_lung: string;
    vit: string;
  };
}
