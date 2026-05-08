import { Box, Container, Typography, Grid, Link } from '@mui/material';
import './Footer.css';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <Box component="footer" sx={{ bgcolor: '#f5f5f5', py: 6, mt: 8 }}>
      <Container maxWidth="lg">
        <Grid container spacing={4} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" gutterBottom>
              LDSC-UNet-ViT
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Advanced lung disease detection and classification using deep learning
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" gutterBottom>
              Resources
            </Typography>
            <Link href="#" display="block" variant="body2" color="textSecondary" sx={{ mb: 1 }}>
              Documentation
            </Link>
            <Link href="#" display="block" variant="body2" color="textSecondary" sx={{ mb: 1 }}>
              GitHub
            </Link>
            <Link href="#" display="block" variant="body2" color="textSecondary">
              License
            </Link>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" gutterBottom>
              Technology
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
              UNet for Segmentation
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
              Vision Transformer (ViT)
            </Typography>
            <Typography variant="body2" color="textSecondary">
              PyTorch + FastAPI
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" gutterBottom>
              Contact
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
              Email: info@ldsc-unet-vit.com
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Support: support@ldsc-unet-vit.com
            </Typography>
          </Grid>
        </Grid>
        <Box sx={{ borderTop: '1px solid #e0e0e0', pt: 3, textAlign: 'center' }}>
          <Typography variant="body2" color="textSecondary">
            &copy; {currentYear} LDSC-UNet-ViT. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}
