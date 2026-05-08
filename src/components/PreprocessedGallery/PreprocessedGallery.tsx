import { Card, CardContent, Grid, Typography, Box, Paper } from '@mui/material';
import { PreprocessingStages } from '../../services/types';
import './PreprocessedGallery.css';

interface PreprocessedGalleryProps {
  stages: PreprocessingStages;
}

export default function PreprocessedGallery({ stages }: PreprocessedGalleryProps) {
  const stageList = [
    { title: 'Original', url: stages.original_url },
    { title: 'Grayscale', url: stages.grayscale_url },
    { title: 'Denoised', url: stages.denoised_url },
    { title: 'Enhanced', url: stages.enhanced_url },
    { title: 'Sharpened', url: stages.sharpened_url },
    { title: 'Normalized', url: stages.normalized_url },
  ];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Preprocessing Stages
        </Typography>
        <Grid container spacing={2}>
          {stageList.map((stage) => (
            <Grid item xs={12} sm={6} md={4} key={stage.title}>
              <Paper sx={{ overflow: 'hidden' }}>
                <img
                  src={stage.url}
                  alt={stage.title}
                  style={{
                    width: '100%',
                    height: 200,
                    objectFit: 'cover',
                  }}
                />
                <Box sx={{ p: 1, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                  <Typography variant="subtitle2">{stage.title}</Typography>
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
}
