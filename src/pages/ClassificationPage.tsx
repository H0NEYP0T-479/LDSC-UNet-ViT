import { useState } from 'react';
import { Container, Box, Alert, CircularProgress, Grid, Typography } from '@mui/material';
import { apiService } from '../services/api';
import { ClassificationResult } from '../services/types';
import UploadCard from '../components/UploadCard/UploadCard';
import PredictionCard from '../components/PredictionCard/PredictionCard';

export default function ClassificationPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);

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
      const response = await apiService.classify(file);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom sx={{ mb: 4 }}>
        Disease Classification
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
            <Grid item xs={12} md={8}>
              <PredictionCard classification={result} />
            </Grid>
          </Grid>
        </Box>
      )}
    </Container>
  );
}
