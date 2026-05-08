import { useState } from 'react';
import { Container, Paper, Box, Alert, CircularProgress, Grid, Card, CardContent, Typography } from '@mui/material';
import { apiService } from '../services/api';
import { SegmentationResult } from '../services/types';
import UploadCard from '../components/UploadCard/UploadCard';
import SegmentationOverlay from '../components/SegmentationOverlay/SegmentationOverlay';

export default function SegmentationPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setResult(null);
  };

  const handleProcess = async () => {
    if (!file) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await apiService.segment(file);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Segmentation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom sx={{ mb: 4 }}>
        Lung Segmentation
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <UploadCard onFileSelect={handleFileSelect} onProcess={handleProcess} loading={loading} />
        </Grid>

        <Grid item xs={12} md={6}>
          {error && <Alert severity="error">{error}</Alert>}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
              <CircularProgress />
            </Box>
          )}
        </Grid>
      </Grid>

      {result && (
        <Box sx={{ mt: 4 }}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <SegmentationOverlay segmentation={result} />
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Segmentation Results
                  </Typography>
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    Disease Detected: <strong>{result.disease_detected ? 'Yes' : 'No'}</strong>
                  </Typography>
                  <Typography variant="body1">
                    Affected Area: <strong>{result.area_percentage.toFixed(2)}%</strong>
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      )}
    </Container>
  );
}
