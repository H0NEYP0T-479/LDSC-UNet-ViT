import { useState } from 'react';
import { Container, Paper, Box, Alert, CircularProgress, Grid, Card, CardContent, Typography } from '@mui/material';
import { apiService } from '../services/api';
import { InferenceResponse } from '../services/types';
import UploadCard from '../components/UploadCard/UploadCard';
import PreprocessedGallery from '../components/PreprocessedGallery/PreprocessedGallery';
import SegmentationOverlay from '../components/SegmentationOverlay/SegmentationOverlay';
import HeatmapOverlay from '../components/HeatmapOverlay/HeatmapOverlay';
import PredictionCard from '../components/PredictionCard/PredictionCard';

export default function InferencePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setResult(null);
  };

  const handleRunInference = async () => {
    if (!file) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await apiService.runFullInference(file);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom sx={{ mb: 4 }}>
        Full Inference Pipeline
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <UploadCard onFileSelect={handleFileSelect} onProcess={handleRunInference} loading={loading} />
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
          <Typography variant="h4" gutterBottom>
            Results
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Processing Time
                  </Typography>
                  <Typography variant="body1">{result.processing_time.toFixed(2)} seconds</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Image ID: {result.image_id}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <PreprocessedGallery stages={result.preprocessing} />
            </Grid>

            <Grid item xs={12} md={6}>
              <SegmentationOverlay segmentation={result.segmentation} />
            </Grid>

            <Grid item xs={12} md={6}>
              <PredictionCard classification={result.classification} />
            </Grid>

            {result.gradcam_url && (
              <Grid item xs={12}>
                <HeatmapOverlay imageUrl={result.gradcam_url} />
              </Grid>
            )}
          </Grid>
        </Box>
      )}
    </Container>
  );
}
