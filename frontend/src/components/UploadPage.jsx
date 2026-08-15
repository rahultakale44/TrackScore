import { useState, useRef } from 'react'
import { uploadVideo } from '../services/api'
import './UploadPage.css'

const SUPPORTED_FORMATS = ['video/mp4', 'video/quicktime', 'video/x-msvideo']
const MAX_FILE_SIZE = 100 * 1024 * 1024 // 100 MB

function UploadPage({ onNavigate, onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [videoPreview, setVideoPreview] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadComplete, setUploadComplete] = useState(false)
  const [videoId, setVideoId] = useState(null)
  const fileInputRef = useRef(null)

  const validateFile = (file) => {
    if (!file) {
      return 'Please select a file'
    }

    if (!SUPPORTED_FORMATS.includes(file.type)) {
      return 'Unsupported file format. Please upload MP4, MOV, or AVI files.'
    }

    if (file.size > MAX_FILE_SIZE) {
      return `File size exceeds 100 MB limit. Selected file: ${(file.size / (1024 * 1024)).toFixed(1)} MB`
    }

    return null
  }

  const handleFileSelect = (file) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setError(null)
    setSelectedFile(file)
    setUploadComplete(false)
    setVideoId(null)

    // Create video preview
    const previewUrl = URL.createObjectURL(file)
    setVideoPreview(previewUrl)
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first')
      return
    }

    setIsUploading(true)
    setError(null)
    setUploadProgress(0)

    try {
      const result = await uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress)
      })

      setUploadComplete(true)
      setVideoId(result.video_id)
      onUploadSuccess(result.video_id)
    } catch (err) {
      console.error('Upload error:', err)
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Upload failed. Please try again.'
      )
      setUploadProgress(0)
    } finally {
      setIsUploading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setVideoPreview(null)
    setUploadProgress(0)
    setIsUploading(false)
    setError(null)
    setUploadComplete(false)
    setVideoId(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="upload-page">
      <header className="upload-header">
        <div className="container">
          <h1 className="logo" onClick={() => onNavigate('home')} style={{ cursor: 'pointer' }}>
            TrackScore
          </h1>
          <nav className="nav">
            <button className="nav-link" onClick={() => onNavigate('home')}>
              Home
            </button>
            <button className="nav-link active">
              Upload
            </button>
          </nav>
        </div>
      </header>

      <main className="upload-main">
        <div className="container">
          <h2 className="page-title">Upload Tennis Video</h2>
          <p className="page-subtitle">
            Upload your tennis match video for comprehensive analysis
          </p>

          <div className="upload-container">
            {!uploadComplete ? (
              <>
                <div
                  className={`dropzone ${isDragging ? 'dragging' : ''}`}
                  onDragEnter={handleDragEnter}
                  onDragLeave={handleDragLeave}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={handleBrowseClick}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/mp4,video/quicktime,video/x-msvideo"
                    onChange={handleFileInputChange}
                    style={{ display: 'none' }}
                  />
                  
                  <div className="dropzone-content">
                    <svg 
                      className="dropzone-icon"
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
                      />
                    </svg>
                    <p className="dropzone-text">
                      {isDragging
                        ? 'Drop your video here'
                        : 'Drag and drop your video here'}
                    </p>
                    <p className="dropzone-subtext">or</p>
                    <button className="browse-button" type="button">
                      Browse Files
                    </button>
                    <p className="dropzone-hint">
                      Supported formats: MP4, MOV, AVI (max 100 MB)
                    </p>
                  </div>
                </div>

                {selectedFile && (
                  <div className="file-info">
                    <div className="file-details">
                      <svg 
                        className="file-icon"
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                      >
                        <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          strokeWidth={2} 
                          d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" 
                        />
                      </svg>
                      <div className="file-text">
                        <p className="file-name">{selectedFile.name}</p>
                        <p className="file-size">{formatFileSize(selectedFile.size)}</p>
                      </div>
                      <button 
                        className="remove-button"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleReset()
                        }}
                        disabled={isUploading}
                      >
                        ✕
                      </button>
                    </div>

                    {videoPreview && (
                      <div className="video-preview">
                        <video 
                          src={videoPreview} 
                          controls 
                          className="preview-video"
                        />
                      </div>
                    )}
                  </div>
                )}

                {isUploading && (
                  <div className="progress-container">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="progress-text">{uploadProgress}% uploaded</p>
                  </div>
                )}

                {error && (
                  <div className="error-message">
                    <svg 
                      className="error-icon"
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
                      />
                    </svg>
                    <p>{error}</p>
                  </div>
                )}

                <div className="action-buttons">
                  <button
                    className="upload-button"
                    onClick={handleUpload}
                    disabled={!selectedFile || isUploading}
                  >
                    {isUploading ? 'Uploading...' : 'Upload Video'}
                  </button>
                  {selectedFile && !isUploading && (
                    <button
                      className="reset-button"
                      onClick={handleReset}
                    >
                      Clear
                    </button>
                  )}
                </div>
              </>
            ) : (
              <div className="success-container">
                <div className="success-icon">✓</div>
                <h3 className="success-title">Upload Successful!</h3>
                <p className="success-message">
                  Your video has been uploaded successfully.
                </p>
                <div className="success-details">
                  <p><strong>Video ID:</strong> {videoId}</p>
                  <p><strong>Filename:</strong> {selectedFile.name}</p>
                  <p><strong>Size:</strong> {formatFileSize(selectedFile.size)}</p>
                </div>
                <div className="success-actions">
                  <button
                    className="upload-button"
                    onClick={handleReset}
                  >
                    Upload Another Video
                  </button>
                  <button
                    className="secondary-button"
                    onClick={() => onNavigate('home')}
                  >
                    Back to Home
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default UploadPage
