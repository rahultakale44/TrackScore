import './Home.css'

function Home({ onNavigate }) {
  return (
    <div className="home">
      <header className="home-header">
        <div className="container">
          <h1 className="logo">TrackScore</h1>
          <nav className="nav">
            <button className="nav-link" onClick={() => onNavigate('home')}>
              Home
            </button>
            <button className="nav-link" onClick={() => onNavigate('upload')}>
              Upload
            </button>
          </nav>
        </div>
      </header>

      <main className="home-main">
        <div className="container">
          <div className="hero">
            <h2 className="hero-title">
              Professional Tennis Video Analysis
            </h2>
            <p className="hero-subtitle">
              Upload your tennis match videos and get comprehensive analytics including player tracking, ball detection, shot classification, and real-time scoring.
            </p>
            <button 
              className="cta-button"
              onClick={() => onNavigate('upload')}
            >
              Get Started
            </button>
          </div>

          <div className="features">
            <div className="feature-card">
              <div className="feature-icon">🎾</div>
              <h3 className="feature-title">Ball Tracking</h3>
              <p className="feature-description">
                Advanced ball detection and trajectory analysis with speed estimation
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">👥</div>
              <h3 className="feature-title">Player Detection</h3>
              <p className="feature-description">
                Real-time player tracking and motion analysis on court
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3 className="feature-title">Shot Classification</h3>
              <p className="feature-description">
                Machine learning powered shot type identification and statistics
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🏆</div>
              <h3 className="feature-title">Live Scoring</h3>
              <p className="feature-description">
                Automatic tennis scoring with point, game, and set tracking
              </p>
            </div>
          </div>
        </div>
      </main>

      <footer className="home-footer">
        <div className="container">
          <p>&copy; 2026 TrackScore. Professional Tennis Analytics.</p>
        </div>
      </footer>
    </div>
  )
}

export default Home
