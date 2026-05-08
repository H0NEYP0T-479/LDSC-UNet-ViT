import React from 'react';
import { Header } from '../components/Header/Header';
import { Footer } from '../components/Footer/Footer';
import { UploadCard } from '../components/UploadCard/UploadCard';
import { ImagePreview } from '../components/ImagePreview/ImagePreview';
import { PreprocessedGallery } from '../components/PreprocessedGallery/PreprocessedGallery';
import { SegmentationOverlay } from '../components/SegmentationOverlay/SegmentationOverlay';
import { PredictionCard } from '../components/PredictionCard/PredictionCard';
import { HeatmapOverlay } from '../components/HeatmapOverlay/HeatmapOverlay';

export const HomePage: React.FC = () => {
  return (
    <div className="home-page">
      <Header />
      <main className="main-content">
        <UploadCard />
        <ImagePreview />
        <PreprocessedGallery />
        <SegmentationOverlay />
        <PredictionCard />
        <HeatmapOverlay />
      </main>
      <Footer />
    </div>
  );
};
