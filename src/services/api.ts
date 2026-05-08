import axios from 'axios'
import type { AxiosInstance } from 'axios'
import type { InferenceResponse, PreprocessingStages, SegmentationResult, ClassificationResult } from './types'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 60000
})

export const checkHealth = async () => {
  const res = await api.get('/api/health')
  return res.data
}

export const runInference = async (file: File): Promise<InferenceResponse> => {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post<InferenceResponse>('/api/inference', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data
}

export const runPreprocessing = async (file: File): Promise<PreprocessingStages> => {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post<PreprocessingStages>('/api/preprocess', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data
}

export const runSegmentation = async (file: File): Promise<SegmentationResult> => {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post<SegmentationResult>('/api/segment', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data
}

export const runClassification = async (file: File): Promise<ClassificationResult> => {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post<ClassificationResult>('/api/classify', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data
}

export default api