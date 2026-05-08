import './ImagePreview.css'

interface Props {
  src: string
  label: string
  baseUrl?: string
}

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function ImagePreview({ src, label, baseUrl = BASE_URL }: Props) {
  const fullSrc = src.startsWith('http') ? src : `${baseUrl}${src}`
  return (
    <div className="image-preview">
      <img src={fullSrc} alt={label} className="preview-img" />
      <span className="preview-label">{label}</span>
    </div>
  )
}