import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';
import HomePage from './pages/HomePage';
import InferencePage from './pages/InferencePage';
import PreprocessingPage from './pages/PreprocessingPage';
import SegmentationPage from './pages/SegmentationPage';
import ClassificationPage from './pages/ClassificationPage';
import HealthPage from './pages/HealthPage';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
import ErrorBoundary from './components/ErrorBoundary/ErrorBoundary';
import './App.css';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <ErrorBoundary>
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <Header />
            <Box sx={{ flex: 1 }}>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/inference" element={<InferencePage />} />
                <Route path="/preprocessing" element={<PreprocessingPage />} />
                <Route path="/segmentation" element={<SegmentationPage />} />
                <Route path="/classification" element={<ClassificationPage />} />
                <Route path="/health" element={<HealthPage />} />
              </Routes>
            </Box>
            <Footer />
          </Box>
        </ErrorBoundary>
      </Router>
    </ThemeProvider>
  );
}

export default App;
