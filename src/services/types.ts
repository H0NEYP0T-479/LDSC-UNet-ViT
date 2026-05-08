export interface PreprocessingStages {
  original_url: string
  grayscale_url: string
  denoised_url: string
  enhanced_url: string
  sharpened_url: string
  normalized_url: string
}

export interface SegmentationResult {
  mask_url: string
  overlay_url: string
  disease_detected: boolean
  area_percentage: number
}

export interface ClassificationResult {
  predicted_class: string
  confidence: number
  probabilities: Record<string, number>
}

export interface InferenceResponse {
  image_id: string
  preprocessing: PreprocessingStages
  segmentation: SegmentationResult
  classification: ClassificationResult
  processing_time: number
}

export type DiseaseClass = 'normal' | 'pneumonia' | 'covid19' | 'tuberculosis'