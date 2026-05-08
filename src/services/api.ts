import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

class APIService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
    });
  }

  async uploadImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    return this.client.post('/preprocessing/process', formData);
  }

  async segmentImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    return this.client.post('/segmentation/segment', formData);
  }

  async classifyImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    return this.client.post('/classification/classify', formData);
  }

  async getHealthStatus() {
    return this.client.get('/health/status');
  }

  async getModelStatus() {
    return this.client.get('/health/models');
  }
}

export const apiService = new APIService();
