import { Card, CardContent, Grid, Typography, Box, Paper } from '@mui/material';
import { SegmentationResult } from '../../services/types';
import './SegmentationOverlay.css';

interface SegmentationOverlayProps {
  segmentation: SegmentationResult;
}

export default function SegmentationOverlay({ segmentation }: SegmentationOverlayProps) {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Lung Segmentation Results
        </Typography>
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6}>
            <Paper sx={{ overflow: 'hidden' }}>
              <img
                src={segmentation.mask_url}
                alt="Segmentation Mask"
                style={{
                  width: '100%',
                  height: 300,
                  objectFit: 'cover',
                }}
              />
              <Box sx={{ p: 1, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                <Typography variant="subtitle2">Segmentation Mask</Typography>
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Paper sx={{ overflow: 'hidden' }}>
              <img
                src={segmentation.overlay_url}
                alt="Overlay"
                style={{
                  width: '100%',
                  height: 300,
                  objectFit: 'cover',
                }}
              />
              <Box sx={{ p: 1, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                <Typography variant="subtitle2">Overlay</Typography>
              </Box>
            </Paper>
          </Grid>
        </Grid>
        <Box sx={{ bgcolor: '#f5f5f5', p: 2, borderRadius: 1 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="textSecondary">
                Disease Detected
              </Typography>
              <Typography variant="h6" sx={{ color: segmentation.disease_detected ? '#d32f2f' : '#388e3c' }}>
                {segmentation.disease_detected ? 'Yes' : 'No'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="textSecondary">
                Affected Area
              </Typography>
              <Typography variant="h6">{segmentation.area_percentage.toFixed(2)}%</Typography>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
}
