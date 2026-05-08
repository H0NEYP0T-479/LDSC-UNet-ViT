import './SegmentationOverlay.css'
import ImagePreview from '../ImagePreview/ImagePreview'
import type { SegmentationResult } from '../../services/types'

interface Props { result: SegmentationResult }

export default function SegmentationOverlay({ result }: Props) {
  return (
    <div className="seg-card">
      <h3 className="seg-title">Segmentation Result</h3>
      <div className="seg-status">
        <span className={`seg-badge ${result.disease_detected ? 'badge-danger' : 'badge-success'}`}>
          {result.disease_detected ? '⚠️ Disease Detected' : '✅ No Disease Detected'}
        </span>
        <span className="seg-area">
          Affected Area: <strong>{result.area_percentage.toFixed(2)}%</strong>
        </span>
      </div>
      <div className="seg-images">
        <ImagePreview src={result.mask_url} label="Segmentation Mask" />
        <ImagePreview src={result.overlay_url} label="Overlay" />
      </div>
    </div>
  )
}