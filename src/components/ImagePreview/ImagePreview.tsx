import React from 'react';
import './ImagePreview.css';

export const ImagePreview: React.FC = () => {
  return (
    <div className="image-preview">
      <h3>Original Image</h3>
      <div className="preview-container">
        <img src="" alt="Preview" className="preview-image" />
      </div>
    </div>
  );
};
