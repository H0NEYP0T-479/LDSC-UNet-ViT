import { Card, CardContent, Typography, Box, LinearProgress, Grid, Chip } from '@mui/material';
import { ClassificationResult } from '../../services/types';
import './PredictionCard.css';

interface PredictionCardProps {
  classification: ClassificationResult;
}

export default function PredictionCard({ classification }: PredictionCardProps) {
  const classColors: Record<string, 'default' | 'error' | 'warning' | 'info' | 'success'> = {
    normal: 'success',
    pneumonia: 'warning',
    covid19: 'error',
    tuberculosis: 'error',
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Disease Classification
        </Typography>

        <Box sx={{ mb: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Predicted Class
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={classification.predicted_class.toUpperCase()}
              color={classColors[classification.predicted_class] || 'default'}
              variant="filled"
            />
            <Box>
              <Typography variant="body2" color="textSecondary">
                Confidence
              </Typography>
              <Typography variant="h6">
                {(classification.confidence * 100).toFixed(2)}%
              </Typography>
            </Box>
          </Box>
        </Box>

        <Typography variant="subtitle2" gutterBottom>
          Class Probabilities
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(classification.probabilities).map(([className, probability]) => (
            <Grid item xs={12} key={className}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" sx={{ minWidth: 100, textTransform: 'capitalize' }}>
                  {className}
                </Typography>
                <Box sx={{ flex: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={probability * 100}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      backgroundColor: '#e0e0e0',
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 4,
                        backgroundColor:
                          probability > 0.5 ? '#d32f2f' : probability > 0.2 ? '#ff9800' : '#388e3c',
                      },
                    }}
                  />
                </Box>
                <Typography variant="body2" sx={{ minWidth: 50, textAlign: 'right' }}>
                  {(probability * 100).toFixed(1)}%
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
}
