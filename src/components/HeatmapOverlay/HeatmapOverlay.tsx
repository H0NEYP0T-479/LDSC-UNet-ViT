import './HeatmapOverlay.css'
import ImagePreview from '../ImagePreview/ImagePreview'

interface Props { overlayUrl: string }

export default function HeatmapOverlay({ overlayUrl }: Props) {
  return (
    <div className="heatmap-card">
      <h3 className="heatmap-title">GradCAM Heatmap</h3>
      <p className="heatmap-desc">
        Red regions indicate areas the ViT model focused on for classification.
      </p>
      <ImagePreview src={overlayUrl} label="GradCAM Overlay" />
    </div>
  )
}