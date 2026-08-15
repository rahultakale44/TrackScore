import { useState } from 'react'
import Home from './components/Home'
import UploadPage from './components/UploadPage'
import './App.css'

function App() {
  const [currentPage, setCurrentPage] = useState('home')

  const handleNavigate = (page) => {
    setCurrentPage(page)
  }

  const handleUploadSuccess = (videoId) => {
    // Store video ID for future use (e.g., navigate to analysis page)
    console.log('Video uploaded successfully:', videoId)
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
    </div>
  )
}

export default App
