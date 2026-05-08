import './PredictionCard.css'
import type { ClassificationResult } from '../../services/types'

const CLASS_COLORS: Record<string, string> = {
  normal: '#22c55e',
  pneumonia: '#f59e0b',
  covid19: '#ef4444',
  tuberculosis: '#a855f7'
}

const CLASS_ICONS: Record<string, string> = {
  normal: '✅',
  pneumonia: '🫁',
  covid19: '🦠',
  tuberculosis: '⚠️'
}

interface Props { result: ClassificationResult }

export default function PredictionCard({ result }: Props) {
  const color = CLASS_COLORS[result.predicted_class] || '#4f8ef7'

  return (
    <div className="prediction-card">
      <h3 className="prediction-title">Classification Result</h3>
      <div className="prediction-main" style={{ borderColor: color }}>
        <span className="prediction-icon">
          {CLASS_ICONS[result.predicted_class] || '🔬'}
        </span>
        <div>
          <div className="prediction-class" style={{ color }}>
            {result.predicted_class.toUpperCase()}
          </div>
          <div className="prediction-confidence">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>
      <div className="prediction-probs">
        {Object.entries(result.probabilities).map(([cls, prob]) => (
          <div key={cls} className="prob-row">
            <span className="prob-label">{cls}</span>
            <div className="prob-bar-bg">
              <div className="prob-bar-fill"
                style={{
                  width: `${(prob * 100).toFixed(1)}%`,
                  background: CLASS_COLORS[cls] || '#4f8ef7'
                }} />
            </div>
            <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}