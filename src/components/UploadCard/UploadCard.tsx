import { useRef, useState } from 'react'
import './UploadCard.css'

interface Props {
  onFileSelect: (file: File) => void
  loading: boolean
}

export default function UploadCard({ onFileSelect, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = (file: File) => {
    setPreview(URL.createObjectURL(file))
    onFileSelect(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div className={`upload-card ${dragOver ? 'drag-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}>
      <input ref={inputRef} type="file" accept="image/*"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        style={{ display: 'none' }} />
      {preview ? (
        <img src={preview} alt="Preview" className="upload-preview" />
      ) : (
        <div className="upload-placeholder">
          <span className="upload-icon">📤</span>
          <p>Drop chest X-ray here or click to upload</p>
          <span className="upload-hint">PNG, JPG, JPEG supported</span>
        </div>
      )}
      {loading && <div className="upload-loading">Analyzing...</div>}
    </div>
  )
}