import { Card, CardContent, Typography, Box, Paper } from '@mui/material';
import './HeatmapOverlay.css';

interface HeatmapOverlayProps {
  imageUrl: string;
}

export default function HeatmapOverlay({ imageUrl }: HeatmapOverlayProps) {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Grad-CAM Heatmap
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
          Visualization of regions contributing to the classification decision
        </Typography>
        <Paper sx={{ overflow: 'hidden' }}>
          <img
            src={imageUrl}
            alt="Grad-CAM Heatmap"
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
            }}
          />
        </Paper>
      </CardContent>
    </Card>
  );
}
