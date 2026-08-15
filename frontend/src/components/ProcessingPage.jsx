import { useState, useEffect, useRef, useCallback } from 'react'
import { startAnalysis, getAnalysisStatus } from '../services/api'
import './ProcessingPage.css'

function ProcessingPage({ videoId, onNavigate, onComplete }) {
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState('idle') // idle, starting, queued, processing, completed, failed
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const [error, setError] = useState(null)
  const pollingInterval = useRef(null)

  const checkStatus = useCallback(async (id) => {
    try {
      const result = await getAnalysisStatus(id)
      setStatus(result.status)
      setProgress(result.progress_percentage)
      setMessage(result.message)

      if (result.status === 'completed') {
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current)
        }
        // Give a short delay to show 100% completion
        setTimeout(() => {
          onComplete(id)
        }, 500)
      } else if (result.status === 'failed') {
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current)
        }
        setError(result.error || 'Analysis failed')
      }
    } catch (err) {
      console.error('Failed to check status:', err)
      // Don't stop polling on temporary network errors
    }
  }, [onComplete])

  const startPolling = useCallback((id) => {
    // Initial check
    checkStatus(id)

    // Poll every 1 second
    pollingInterval.current = setInterval(() => {
      checkStatus(id)
    }, 1000)
  }, [checkStatus])

  useEffect(() => {
    const initiateAnalysis = async () => {
      if (!videoId || jobId) return

      setStatus('starting')
      setError(null)

      try {
        const result = await startAnalysis(videoId)
        setJobId(result.job_id)
        setStatus(result.status)
        startPolling(result.job_id)
      } catch (err) {
        console.error('Failed to start analysis:', err)
        setError(err.response?.data?.detail || 'Failed to start analysis')
        setStatus('failed')
      }
    }

    initiateAnalysis()
  }, [videoId, jobId, startPolling])

  useEffect(() => {
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current)
      }
    }
  }, [])

  const getStatusIcon = () => {
    switch (status) {
      case 'queued':
        return '⏳'
      case 'processing':
        return '⚙️'
      case 'completed':
        return '✓'
      case 'failed':
        return '✕'
      default:
        return '⏳'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'idle':
        return 'Initializing...'
      case 'starting':
        return 'Starting analysis...'
      case 'queued':
        return 'Queued for processing'
      case 'processing':
        return 'Processing video'
      case 'completed':
        return 'Analysis complete!'
      case 'failed':
        return 'Analysis failed'
      default:
        return 'Processing...'
    }
  }

  return (
    <div className="processing-page">
      <header className="processing-header">
        <div className="container">
          <h1 className="logo" onClick={() => onNavigate('home')} style={{ cursor: 'pointer' }}>
            TrackScore
          </h1>
        </div>
      </header>

      <main className="processing-main">
        <div className="container">
          <div className="processing-container">
            <div className={`status-icon ${status}`}>
              {getStatusIcon()}
            </div>

            <h2 className="status-title">{getStatusText()}</h2>

            {message && (
              <p className="status-message">{message}</p>
            )}

            {(status === 'queued' || status === 'processing' || status === 'starting') && (
              <div className="progress-container">
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="progress-text">{Math.round(progress)}%</p>
              </div>
            )}

            {error && (
              <div className="error-box">
                <p className="error-title">Error Details</p>
                <p className="error-text">{error}</p>
                <button 
                  className="retry-button"
                  onClick={() => onNavigate('upload')}
                >
                  Upload Another Video
                </button>
              </div>
            )}

            {status === 'completed' && (
              <div className="completion-message">
                <p>Redirecting to results...</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default ProcessingPage
