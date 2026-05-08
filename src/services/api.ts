import axios, { AxiosInstance } from 'axios';
import {
  PreprocessingResult,
  SegmentationResult,
  ClassificationResult,
  InferenceResponse,
  HealthResponse,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

class APIService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 60000,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  async getHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  async preprocess(file: File): Promise<PreprocessingResult> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await this.client.post<PreprocessingResult>('/preprocess', formData);
    return response.data;
  }

  async segment(file: File): Promise<SegmentationResult> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await this.client.post<SegmentationResult>('/segment', formData);
    return response.data;
  }

  async classify(file: File): Promise<ClassificationResult> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await this.client.post<ClassificationResult>('/classify', formData);
    return response.data;
  }

  async runFullInference(file: File): Promise<InferenceResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await this.client.post<InferenceResponse>('/inference', formData);
    return response.data;
  }
}

export const apiService = new APIService();
