import { useState } from 'react'
import UploadCard from '../components/UploadCard/UploadCard'
import PredictionCard from '../components/PredictionCard/PredictionCard'
import SegmentationOverlay from '../components/SegmentationOverlay/SegmentationOverlay'
import PreprocessedGallery from '../components/PreprocessedGallery/PreprocessedGallery'
import type { InferenceResponse } from '../services/types'
import { runInference } from '../services/api'
import './HomePage.css'

export default function HomePage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<InferenceResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileSelect = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await runInference(file)
      setResult(data)
    } catch (e: any) {
      setError(e.message || 'Inference failed. Make sure backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="home">
      <div className="home-hero">
        <h1 className="home-title">
          Lung Disease <span className="highlight">Detection</span>
        </h1>
        <p className="home-subtitle">
          Upload a chest X-ray to classify and segment lung disease using
          Vision Transformer + UNet deep learning models.
        </p>
      </div>

      <div className="home-upload">
        <UploadCard onFileSelect={handleFileSelect} loading={loading} />
      </div>

      {error && (
        <div className="home-error">⚠️ {error}</div>
      )}

      {result && (
        <div className="home-results">
          <div className="results-meta">
            <span>🆔 {result.image_id}</span>
            <span>⏱ {result.processing_time}s</span>
          </div>

          <div className="results-grid">
            <PredictionCard result={result.classification} />
            <SegmentationOverlay result={result.segmentation} />
          </div>

          <PreprocessedGallery stages={result.preprocessing} />
        </div>
      )}
    </div>
  )
}