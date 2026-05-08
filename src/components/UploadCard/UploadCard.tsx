import React, { useState } from 'react';
import './UploadCard.css';
import { apiService } from '../../services/api';

export const UploadCard: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await apiService.uploadImage(file);
      console.log('Upload successful:', response.data);
    } catch (err) {
      setError('Failed to upload image. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="upload-card">
      <h2>Upload Medical Image</h2>
      <div className="upload-area">
        <label htmlFor="file-input" className="upload-label">
          <div className="upload-icon">📁</div>
          <p>Click to upload or drag and drop</p>
          <p className="upload-hint">PNG, JPG, DICOM up to 50MB</p>
        </label>
        <input
          id="file-input"
          type="file"
          onChange={handleFileUpload}
          accept=".png,.jpg,.jpeg,.dcm"
          disabled={isLoading}
          className="file-input"
        />
      </div>
      {isLoading && <p className="loading">Uploading...</p>}
      {error && <p className="error">{error}</p>}
    </div>
  );
};
