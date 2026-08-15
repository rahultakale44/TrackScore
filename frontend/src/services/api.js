import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const healthCheck = async () => {
  const response = await api.get('/api/health')
  return response.data
}

export const uploadVideo = async (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/api/videos/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        )
        onProgress(percentCompleted)
      }
    },
  })

  return response.data
}

export const startAnalysis = async (videoId, maxFrames = null) => {
  const response = await api.post(`/api/analysis/start/${videoId}`, {
    max_frames: maxFrames,
  })
  return response.data
}

export const getAnalysisStatus = async (jobId) => {
  const response = await api.get(`/api/analysis/status/${jobId}`)
  return response.data
}

export const getAnalysisResult = async (jobId) => {
  const response = await api.get(`/api/analysis/result/${jobId}`)
  return response.data
}

export const getAnalysisMetadata = async (jobId) => {
  const response = await api.get(`/api/analysis/metadata/${jobId}`)
  return response.data
}

export default api
