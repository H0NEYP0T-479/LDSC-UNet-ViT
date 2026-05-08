# LDSC-UNet-ViT Frontend

Modern React + TypeScript frontend for the Lung Disease Segmentation and Classification system.

## Features

- **Full Inference Pipeline**: Complete analysis with preprocessing, segmentation, and classification
- **Image Preprocessing**: Visualize 6 stages of medical image preprocessing
- **Lung Segmentation**: UNet-based lung segmentation with overlay
- **Disease Classification**: Vision Transformer-based disease classification
- **System Health**: Check backend and model status in real-time
- **Material-UI**: Professional UI components
- **Responsive Design**: Works on desktop, tablet, and mobile

## Structure

```
src/
├── pages/           # Page components for each feature
│   ├── HomePage.tsx
│   ├── InferencePage.tsx
│   ├── PreprocessingPage.tsx
│   ├── SegmentationPage.tsx
│   ├── ClassificationPage.tsx
│   └── HealthPage.tsx
├── components/      # Reusable UI components
│   ├── Header/
│   ├── Footer/
│   ├── UploadCard/
│   ├── PreprocessedGallery/
│   ├── SegmentationOverlay/
│   ├── PredictionCard/
│   ├── HeatmapOverlay/
│   ├── ImagePreview/
│   └── ErrorBoundary/
├── services/        # API integration
│   ├── api.ts       # API client
│   └── types.ts     # TypeScript types
└── App.tsx          # Main app with routing

```

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

3. Update `VITE_API_URL` if backend is not on localhost:8000

4. Start development server:
```bash
npm run dev
```

5. Build for production:
```bash
npm run build
```

## API Integration

The frontend connects to the backend FastAPI server at:
- Default: `http://localhost:8000/api`
- Configurable via `VITE_API_URL` environment variable

### Available Endpoints

- `GET /health` - System health check
- `POST /preprocess` - Image preprocessing
- `POST /segment` - Lung segmentation
- `POST /classify` - Disease classification
- `POST /inference` - Full pipeline inference

## Pages

### Home Page
- Overview of all features
- Quick navigation to different analysis tools

### Full Inference
- Complete analysis pipeline
- Shows all preprocessing stages, segmentation, classification, and Grad-CAM

### Image Preprocessing
- 6 preprocessing stages: Original, Grayscale, Denoised, Enhanced, Sharpened, Normalized
- Useful for understanding image preparation

### Lung Segmentation
- UNet-based segmentation results
- Shows mask, overlay, and area percentage
- Indicates if abnormality detected

### Disease Classification
- Vision Transformer classification
- Shows predicted class, confidence, and per-class probabilities

### System Health
- Backend connectivity status
- Model loading status (ViT and UNet)
- Auto-refreshes every 10 seconds

## Technologies

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Material-UI** - Component library
- **React Router** - Navigation
- **Axios** - HTTP client

## Environment Variables

```bash
VITE_API_URL=http://localhost:8000/api
```

## Development

- HMR (Hot Module Replacement) enabled
- Strict mode for React development
- Error boundaries for error handling
- Responsive Material-UI theme

## Performance

- Lazy loading for pages
- Image optimization
- API response caching
- Timeout handling (60s for inference)

## Error Handling

- Error boundary for catch-all errors
- HTTP error handling in API service
- User-friendly error messages
- Retry mechanisms

## Future Enhancements

- [ ] Batch processing
- [ ] Historical results
- [ ] User authentication
- [ ] Result export (PDF, CSV)
- [ ] Real-time processing progress
- [ ] Image annotation tools
- [ ] Model performance metrics
- [ ] API documentation (Swagger UI)
