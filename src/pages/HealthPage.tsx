import { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  CircularProgress,
  Alert,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { apiService } from '../services/api';
import { HealthResponse } from '../services/types';

export default function HealthPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const response = await apiService.getHealth();
      setHealth(response);
      setError(null);
    } catch (err: any) {
      setError('Failed to connect to backend');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom sx={{ mb: 4 }}>
        System Status
      </Typography>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      )}

      {error && <Alert severity="error">{error}</Alert>}

      {health && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Typography variant="h6">Overall Status:</Typography>
                  <Chip
                    icon={health.status === 'healthy' ? <CheckCircleIcon /> : <ErrorIcon />}
                    label={health.status.toUpperCase()}
                    color={health.status === 'healthy' ? 'success' : 'warning'}
                    variant="outlined"
                  />
                </Box>
                <Typography variant="body2" color="textSecondary">
                  API Version: {health.version}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Model Status
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(health.models).map(([modelName, status]) => (
                <Grid item xs={12} sm={6} md={3} key={modelName}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="textSecondary">
                        {modelName.toUpperCase()}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                        <Chip
                          icon={status === 'loaded' ? <CheckCircleIcon /> : <ErrorIcon />}
                          label={status}
                          color={status === 'loaded' ? 'success' : 'error'}
                          size="small"
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Grid>
        </Grid>
      )}
    </Container>
  );
}
