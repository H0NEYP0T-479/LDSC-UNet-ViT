import { AppBar, Toolbar, Typography, Box, Button, Container } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import './Header.css';

export default function Header() {
  const navItems = [
    { label: 'Home', path: '/' },
    { label: 'Full Analysis', path: '/inference' },
    { label: 'Preprocessing', path: '/preprocessing' },
    { label: 'Segmentation', path: '/segmentation' },
    { label: 'Classification', path: '/classification' },
    { label: 'Health', path: '/health' },
  ];

  return (
    <AppBar position="sticky">
      <Container maxWidth="lg">
        <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FavoriteBorderIcon />
            <Typography
              variant="h6"
              component={RouterLink}
              to="/"
              sx={{
                textDecoration: 'none',
                color: 'inherit',
                fontWeight: 'bold',
              }}
            >
              LDSC-UNet-ViT
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {navItems.map((item) => (
              <Button
                key={item.path}
                color="inherit"
                component={RouterLink}
                to={item.path}
                sx={{
                  textTransform: 'none',
                  fontSize: '0.9rem',
                  '&:hover': { backgroundColor: 'rgba(255,255,255,0.1)' },
                }}
              >
                {item.label}
              </Button>
            ))}
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}
