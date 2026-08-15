import { useState, useEffect } from 'react'
import { getAnalysisResult, getAnalysisMetadata } from '../services/api'
import './DashboardPage.css'

function DashboardPage({ jobId, onNavigate }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [metadata, setMetadata] = useState(null)
  const [warnings, setWarnings] = useState([])

  useEffect(() => {
    if (jobId) {
      const fetchData = async () => {
        setLoading(true)
        setError(null)

        try {
          const [resultData, metadataData] = await Promise.all([
            getAnalysisResult(jobId),
            getAnalysisMetadata(jobId)
          ])

          setResult(resultData.summary)
          setMetadata(metadataData.metadata)
          setWarnings([...resultData.warnings, ...metadataData.warnings])
        } catch (err) {
          console.error('Failed to fetch analysis data:', err)
          setError(err.response?.data?.detail || 'Failed to load analysis results')
        } finally {
          setLoading(false)
        }
      }

      fetchData()
    }
  }, [jobId])

  const getVideoUrl = () => {
    const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
    return `${baseUrl}/api/analysis/video/${jobId}`
  }

  const handleDownloadVideo = () => {
    window.open(getVideoUrl(), '_blank')
  }

  if (loading) {
    return (
      <div className="dashboard-page">
        <header className="dashboard-header">
          <div className="container">
            <h1 className="logo" onClick={() => onNavigate('home')} style={{ cursor: 'pointer' }}>
              TrackScore
            </h1>
          </div>
        </header>
        <main className="dashboard-main">
          <div className="container">
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Loading analytics...</p>
            </div>
          </div>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="dashboard-page">
        <header className="dashboard-header">
          <div className="container">
            <h1 className="logo" onClick={() => onNavigate('home')} style={{ cursor: 'pointer' }}>
              TrackScore
            </h1>
          </div>
        </header>
        <main className="dashboard-main">
          <div className="container">
            <div className="error-container">
              <div className="error-icon">✕</div>
              <h2>Failed to Load Analytics</h2>
              <p>{error}</p>
              <button className="action-button" onClick={() => onNavigate('home')}>
                Back to Home
              </button>
            </div>
          </div>
        </main>
      </div>
    )
  }

  const video = result?.video || {}
  const processing = result?.processing || {}
  const detections = result?.detections || {}

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <div className="container">
          <h1 className="logo" onClick={() => onNavigate('home')} style={{ cursor: 'pointer' }}>
            TrackScore
          </h1>
          <div className="header-actions">
            <button className="nav-button" onClick={() => onNavigate('upload')}>
              New Upload
            </button>
          </div>
        </div>
      </header>

      <main className="dashboard-main">
        <div className="container">
          <div className="dashboard-title-section">
            <h2 className="dashboard-title">Tennis Video Analysis</h2>
            <button className="download-button" onClick={handleDownloadVideo}>
              📥 Download Analytics Video
            </button>
          </div>

          {warnings.length > 0 && (
            <div className="warnings-box">
              <p className="warnings-title">⚠️ Warnings</p>
              {warnings.map((warning, index) => (
                <p key={index} className="warning-item">{warning}</p>
              ))}
            </div>
          )}

          {/* Video Information */}
          <section className="dashboard-section">
            <h3 className="section-title">Video Information</h3>
            <div className="info-grid">
              <div className="info-card">
                <div className="info-label">Filename</div>
                <div className="info-value">{video.filename || 'Not available'}</div>
              </div>
              <div className="info-card">
                <div className="info-label">Resolution</div>
                <div className="info-value">{video.resolution || 'Not available'}</div>
              </div>
              <div className="info-card">
                <div className="info-label">FPS</div>
                <div className="info-value">{video.fps ? `${video.fps} fps` : 'Not available'}</div>
              </div>
              <div className="info-card">
                <div className="info-label">Duration</div>
                <div className="info-value">
                  {video.duration_seconds ? `${video.duration_seconds.toFixed(1)}s` : 'Not available'}
                </div>
              </div>
              <div className="info-card">
                <div className="info-label">Total Frames</div>
                <div className="info-value">{video.frame_count || 'Not available'}</div>
              </div>
              <div className="info-card">
                <div className="info-label">Processed Frames</div>
                <div className="info-value">{processing.frames_processed || 'Not available'}</div>
              </div>
            </div>
          </section>

          {/* Detection Statistics */}
          <section className="dashboard-section">
            <h3 className="section-title">Detection Statistics</h3>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-icon">👥</div>
                <div className="stat-label">Player Detections</div>
                <div className="stat-value">
                  {detections.players?.total_detections ?? 'Not available'}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-icon">🎾</div>
                <div className="stat-label">Ball Detections</div>
                <div className="stat-value">
                  {detections.ball?.total_detections ?? 'Not available'}
                </div>
              </div>
            </div>
          </section>

          {/* Scoreboard (if available in metadata) */}
          {metadata?.pipeline_result?.scoring && (
            <section className="dashboard-section">
              <h3 className="section-title">Match Scoreboard</h3>
              <div className="scoreboard">
                <div className="scoreboard-row">
                  <div className="player-name">Player A</div>
                  <div className="score-box">
                    {metadata.pipeline_result.scoring.points?.['Player A'] || '0'}
                  </div>
                  <div className="score-box">
                    {metadata.pipeline_result.scoring.games?.['Player A'] || '0'}
                  </div>
                  <div className="score-box">
                    {metadata.pipeline_result.scoring.sets?.['Player A'] || '0'}
                  </div>
                </div>
                <div className="scoreboard-row">
                  <div className="player-name">Player B</div>
                  <div className="score-box">
                    {metadata.pipeline_result.scoring.points?.['Player B'] || '0'}
                  </div>
                  <div className="score-box">
                    {metadata.pipeline_result.scoring.games?.['Player B'] || '0'}
                  </div>
                  <div className="score-box">
                    {metadata.pipeline_result.scoring.sets?.['Player B'] || '0'}
                  </div>
                </div>
                <div className="scoreboard-labels">
                  <span>Player</span>
                  <span>Points</span>
                  <span>Games</span>
                  <span>Sets</span>
                </div>
                {metadata.pipeline_result.scoring.winner && (
                  <div className="winner-banner">
                    🏆 Winner: {metadata.pipeline_result.scoring.winner}
                  </div>
                )}
                {metadata.pipeline_result.scoring.leader && !metadata.pipeline_result.scoring.winner && (
                  <div className="leader-info">
                    Leading: {metadata.pipeline_result.scoring.leader}
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Analytics Cards */}
          {metadata?.pipeline_result?.analytics && (
            <section className="dashboard-section">
              <h3 className="section-title">Advanced Analytics</h3>
              <div className="analytics-grid">
                {metadata.pipeline_result.analytics.player_distance && (
                  <div className="analytics-card">
                    <div className="analytics-label">Player Distance</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.player_distance.toFixed(2)} m
                    </div>
                  </div>
                )}
                {metadata.pipeline_result.analytics.player_speed && (
                  <div className="analytics-card">
                    <div className="analytics-label">Player Speed</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.player_speed.toFixed(2)} km/h
                    </div>
                  </div>
                )}
                {metadata.pipeline_result.analytics.ball_speed && (
                  <div className="analytics-card">
                    <div className="analytics-label">Ball Speed</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.ball_speed.toFixed(2)} km/h
                    </div>
                  </div>
                )}
                {metadata.pipeline_result.analytics.rallies !== undefined && (
                  <div className="analytics-card">
                    <div className="analytics-label">Rallies</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.rallies}
                    </div>
                  </div>
                )}
                {metadata.pipeline_result.analytics.shots !== undefined && (
                  <div className="analytics-card">
                    <div className="analytics-label">Shots</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.shots}
                    </div>
                  </div>
                )}
                {metadata.pipeline_result.analytics.bounces !== undefined && (
                  <div className="analytics-card">
                    <div className="analytics-label">Bounces</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.bounces}
                    </div>
                  </div>
                )}
                {metadata.pipeline_result.analytics.in_count !== undefined && (
                  <div className="analytics-card">
                    <div className="analytics-label">IN Count</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.in_count}
                    </div>
                  </div>
                )}
                {metadata.pipeline_result.analytics.out_count !== undefined && (
                  <div className="analytics-card">
                    <div className="analytics-label">OUT Count</div>
                    <div className="analytics-value">
                      {metadata.pipeline_result.analytics.out_count}
                    </div>
                  </div>
                )}
              </div>
              {!metadata.pipeline_result.analytics && (
                <div className="not-available-message">
                  Advanced analytics not available for this video
                </div>
              )}
            </section>
          )}

          {/* Event Timeline */}
          {metadata?.pipeline_result?.events && metadata.pipeline_result.events.length > 0 && (
            <section className="dashboard-section">
              <h3 className="section-title">Event Timeline</h3>
              <div className="timeline">
                {metadata.pipeline_result.events.map((event, index) => (
                  <div key={index} className="timeline-item">
                    <div className="timeline-time">
                      {event.timestamp ? `${event.timestamp.toFixed(2)}s` : 'N/A'}
                    </div>
                    <div className="timeline-content">
                      <div className="timeline-type">{event.type || 'Event'}</div>
                      <div className="timeline-description">
                        {event.description || 'No description'}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          <div className="dashboard-actions">
            <button className="action-button" onClick={() => onNavigate('upload')}>
              Analyze Another Video
            </button>
            <button className="secondary-action-button" onClick={() => onNavigate('home')}>
              Back to Home
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

export default DashboardPage
