import { useState } from 'react'
import Home from './components/Home'
import UploadPage from './components/UploadPage'
import ProcessingPage from './components/ProcessingPage'
import DashboardPage from './components/DashboardPage'
import './App.css'

function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [videoId, setVideoId] = useState(null)
  const [jobId, setJobId] = useState(null)

  const handleNavigate = (page) => {
    setCurrentPage(page)
  }

  const handleUploadSuccess = (uploadedVideoId) => {
    setVideoId(uploadedVideoId)
    setCurrentPage('processing')
  }

  const handleProcessingComplete = (completedJobId) => {
    setJobId(completedJobId)
    setCurrentPage('dashboard')
  }

  return (
    <div className="app">
      {currentPage === 'home' && <Home onNavigate={handleNavigate} />}
      {currentPage === 'upload' && (
        <UploadPage 
          onNavigate={handleNavigate} 
          onUploadSuccess={handleUploadSuccess}
        />
      )}
      {currentPage === 'processing' && (
        <ProcessingPage
          videoId={videoId}
          onNavigate={handleNavigate}
          onComplete={handleProcessingComplete}
        />
      )}
      {currentPage === 'dashboard' && (
        <DashboardPage
          jobId={jobId}
          onNavigate={handleNavigate}
        />
      )}
    </div>
  )
}

export default App
