import { useState } from 'react';
import {
  Card,
  CardContent,
  Box,
  Button,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import './UploadCard.css';

interface UploadCardProps {
  onFileSelect: (file: File) => void;
  onProcess: () => void;
  loading: boolean;
}

export default function UploadCard({ onFileSelect, onProcess, loading }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    setFile(selectedFile);
    setError(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      setPreview(event.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);

    onFileSelect(selectedFile);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      const input = document.createElement('input');
      input.type = 'file';
      const dt = new DataTransfer();
      dt.items.add(droppedFile);
      const inputEvent = new Event('change', { bubbles: true });
      Object.defineProperty(inputEvent, 'target', { writable: false, value: { files: dt.files } });
      handleFileChange(inputEvent as any);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Upload Medical Image
        </Typography>

        {error && <Alert severity="error">{error}</Alert>}

        <Box
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          sx={{
            border: '2px dashed #ccc',
            borderRadius: 1,
            p: 3,
            textAlign: 'center',
            mb: 2,
            cursor: 'pointer',
            transition: 'all 0.3s',
            '&:hover': {
              borderColor: '#1976d2',
              bgcolor: '#f5f5f5',
            },
          }}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="file-input"
            disabled={loading}
          />
          <label htmlFor="file-input" style={{ cursor: 'pointer', display: 'block' }}>
            <CloudUploadIcon sx={{ fontSize: 48, color: '#1976d2', mb: 1 }} />
            <Typography variant="body1">Click to upload or drag and drop</Typography>
            <Typography variant="caption" color="textSecondary">
              PNG, JPG, DICOM up to 50MB
            </Typography>
          </label>
        </Box>

        {preview && (
          <Box sx={{ mb: 2 }}>
            <img
              src={preview}
              alt="preview"
              style={{
                maxWidth: '100%',
                maxHeight: 300,
                borderRadius: 4,
              }}
            />
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              {file?.name}
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          fullWidth
          onClick={onProcess}
          disabled={!file || loading}
          startIcon={loading ? <CircularProgress size={20} /> : undefined}
        >
          {loading ? 'Processing...' : 'Process Image'}
        </Button>
      </CardContent>
    </Card>
  );
}

