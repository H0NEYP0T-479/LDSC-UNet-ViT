import React from 'react';
import './PreprocessedGallery.css';

export const PreprocessedGallery: React.FC = () => {
  return (
    <div className="gallery">
      <h3>Preprocessing Results</h3>
      <div className="gallery-grid">
        <div className="gallery-item">
          <p>Original</p>
        </div>
        <div className="gallery-item">
          <p>Contrast Enhanced</p>
        </div>
        <div className="gallery-item">
          <p>CLAHE</p>
        </div>
        <div className="gallery-item">
          <p>Denoised</p>
        </div>
      </div>
    </div>
  );
};
