import React from 'react';
import './SegmentationOverlay.css';

export const SegmentationOverlay: React.FC = () => {
  return (
    <div className="segmentation-overlay">
      <h3>Lung Segmentation Results</h3>
      <div className="overlay-container">
        <div className="original">
          <p>Original</p>
        </div>
        <div className="mask">
          <p>Segmentation Mask</p>
        </div>
        <div className="overlay">
          <p>Overlay</p>
        </div>
      </div>
    </div>
  );
};
