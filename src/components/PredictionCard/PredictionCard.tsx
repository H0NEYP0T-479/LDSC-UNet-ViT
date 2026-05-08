import React from 'react';
import './PredictionCard.css';

export const PredictionCard: React.FC = () => {
  return (
    <div className="prediction-card">
      <h3>Disease Classification Results</h3>
      <div className="prediction-results">
        <div className="prediction-item">
          <span className="label">Prediction:</span>
          <span className="value">-</span>
        </div>
        <div className="prediction-item">
          <span className="label">Confidence:</span>
          <span className="value">-</span>
        </div>
      </div>
      <div className="probabilities">
        <h4>Class Probabilities</h4>
        <div className="prob-list">
          <div className="prob-item">
            <span>Normal</span>
            <div className="prob-bar"><div className="prob-fill"></div></div>
          </div>
          <div className="prob-item">
            <span>Pneumonia</span>
            <div className="prob-bar"><div className="prob-fill"></div></div>
          </div>
          <div className="prob-item">
            <span>Tuberculosis</span>
            <div className="prob-bar"><div className="prob-fill"></div></div>
          </div>
          <div className="prob-item">
            <span>COVID-19</span>
            <div className="prob-bar"><div className="prob-fill"></div></div>
          </div>
        </div>
      </div>
    </div>
  );
};
