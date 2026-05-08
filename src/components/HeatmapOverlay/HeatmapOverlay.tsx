import React from 'react';
import './HeatmapOverlay.css';

export const HeatmapOverlay: React.FC = () => {
  return (
    <div className="heatmap-overlay">
      <h3>Grad-CAM Heatmap</h3>
      <p className="description">
        Visualization of regions contributing to the classification decision
      </p>
      <div className="heatmap-container">
        <div className="heatmap-item">
          <p>Original Image</p>
        </div>
        <div className="heatmap-item">
          <p>Heatmap</p>
        </div>
        <div className="heatmap-item">
          <p>Overlay</p>
        </div>
      </div>
    </div>
  );
};
