import './PreprocessedGallery.css'
import ImagePreview from '../ImagePreview/ImagePreview'
import type { PreprocessingStages } from '../../services/types'

interface Props { stages: PreprocessingStages }

export default function PreprocessedGallery({ stages }: Props) {
  const items = [
    { src: stages.original_url, label: 'Original' },
    { src: stages.grayscale_url, label: 'Grayscale' },
    { src: stages.denoised_url, label: 'Denoised' },
    { src: stages.enhanced_url, label: 'CLAHE Enhanced' },
    { src: stages.sharpened_url, label: 'Sharpened' },
    { src: stages.normalized_url, label: 'Normalized' }
  ]

  return (
    <div className="gallery-card">
      <h3 className="gallery-title">Preprocessing Pipeline</h3>
      <div className="gallery-grid">
        {items.map((item) => (
          <ImagePreview key={item.label} src={item.src} label={item.label} />
        ))}
      </div>
    </div>
  )
}