import { Container, Grid, Card, CardContent, CardActions, Button, Typography, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import ImageIcon from '@mui/icons-material/Image';
import MergeIcon from '@mui/icons-material/Merge';
import PsychologyIcon from '@mui/icons-material/Psychology';
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';

export default function HomePage() {
  const navigate = useNavigate();

  const features = [
    {
      title: 'Full Inference',
      description: 'Run complete analysis: preprocessing, segmentation, and classification',
      icon: <FavoriteBorderIcon sx={{ fontSize: 40, color: '#1976d2' }} />,
      path: '/inference',
      color: 'primary',
    },
    {
      title: 'Image Preprocessing',
      description: 'See 6 stages of medical image preprocessing',
      icon: <ImageIcon sx={{ fontSize: 40, color: '#f57c00' }} />,
      path: '/preprocessing',
      color: 'warning',
    },
    {
      title: 'Lung Segmentation',
      description: 'Segment lungs from chest X-rays using UNet',
      icon: <MergeIcon sx={{ fontSize: 40, color: '#388e3c' }} />,
      path: '/segmentation',
      color: 'success',
    },
    {
      title: 'Disease Classification',
      description: 'Classify diseases using Vision Transformer',
      icon: <PsychologyIcon sx={{ fontSize: 40, color: '#d32f2f' }} />,
      path: '/classification',
      color: 'error',
    },
    {
      title: 'System Health',
      description: 'Check backend and model status',
      icon: <FitnessCenterIcon sx={{ fontSize: 40, color: '#7b1fa2' }} />,
      path: '/health',
      color: 'secondary',
    },
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Box sx={{ textAlign: 'center', mb: 8 }}>
        <Typography variant="h2" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
          LDSC-UNet-ViT
        </Typography>
        <Typography variant="h5" color="textSecondary" sx={{ mb: 4 }}>
          Lung Disease Segmentation and Classification
        </Typography>
        <Typography variant="body1" color="textSecondary" sx={{ maxWidth: 600, mx: 'auto' }}>
          Advanced medical image analysis system using Deep Learning. Combines UNet for precise lung segmentation and Vision Transformer for accurate disease classification.
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'all 0.3s',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: 6,
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', pb: 1 }}>
                <Box sx={{ mb: 2 }}>{feature.icon}</Box>
                <Typography gutterBottom variant="h6" component="h3">
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'center', pb: 2 }}>
                <Button
                  variant="contained"
                  color={feature.color as any}
                  size="small"
                  onClick={() => navigate(feature.path)}
                >
                  Open
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}
