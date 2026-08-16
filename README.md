# TrackScore

**CLI Tennis Video Analytics** powered by Machine Learning and Computer Vision

TrackScore is a command-line Python system for analyzing tennis match videos, providing automated player and ball tracking, bounce detection, shot classification, and real-time scoring overlays. Built with OpenCV and YOLO11, it processes videos directly from the command line to generate annotated match footage with comprehensive analytics.

---

## Overview

TrackScore automates tennis match analysis by:
- Detecting and tracking players throughout the match
- Detecting and tracking the tennis ball with trajectory prediction
- Identifying court lines and establishing court geometry
- Detecting ball bounces and classifying IN/OUT calls
- Estimating ball and player speeds (court-plane approximations)
- Classifying shot types using machine learning
- Tracking live match scores (points, games, sets)
- Rendering professional video overlays with analytics

** Limitations & Disclaimers:**
- Ball speed estimates are court-plane projections from monocular video, not true 3D velocities
- Bounce detection and IN/OUT classification depend on tracking quality and court calibration accuracy
- Shot classifier requires labeled training data; demo data is synthetic and not representative of production accuracy
- This is a portfolio/research project, **not a certified electronic line-calling system**
- Accuracy varies significantly with video quality, camera angle, and lighting conditions

---

##  Quick Start

```bash
# Basic usage
python trackscore.py samples/tennis_match.mp4

# Process first 10 seconds only
python trackscore.py samples/tennis_match.mp4 --max-seconds 10

# Disable preview window
python trackscore.py samples/tennis_match.mp4 --no-preview

# Custom output location
python trackscore.py video.mp4 --output outputs/my_analysis.mp4
```

**Output:**
- `outputs/final/trackscore_analysis.mp4` - Annotated video with overlays
- `outputs/final/analytics.json` - Complete frame-by-frame analytics
- `outputs/final/summary.json` - Processing summary and statistics

---

##  Features

### Computer Vision Pipeline
- **Player Detection & Tracking**: YOLO11-based person detection with stable ID assignment
- **Ball Detection & Tracking**: High-frequency ball detection with Kalman-based trajectory prediction
- **Court Analysis**: Automated court line detection and homography calibration
- **Bounce Detection**: Trajectory analysis for bounce identification with spatial validation
- **Speed Estimation**: Frame-by-frame speed calculation for players and ball (estimated, not certified)

### Machine Learning
- **Shot Classification**: Multi-model ensemble (Random Forest, Gradient Boosting, SVM) for shot type prediction
- **Feature Extraction**: Automated extraction of temporal, spatial, and motion features from rallies
- **Training Pipeline**: Configurable training with synthetic demo datasets

### Match Analytics
- **Live Scoring**: Automatic point, game, and set tracking following tennis rules
- **Event Detection**: Rally segmentation, shot detection, bounce calls
- **Statistics**: Player distance covered, shot counts, rally lengths, IN/OUT statistics
- **Performance Metrics**: Speed profiles, court coverage analysis

### Video Rendering
- **Professional Overlays**: TrackScore branding, live scoreboard, player labels, ball markers
- **Trajectory Visualization**: 30-frame ball trajectory tail with predicted vs detected distinction
- **Event Annotations**: BOUNCE, IN, OUT, SHOT, RALLY overlays with fade effects
- **Resolution-Adaptive**: Automatic font and element scaling based on video resolution
- **Live Preview**: OpenCV window during processing (press 'q' to quit)

---

##  System Architecture

```mermaid
graph TB
    subgraph Frontend["React Frontend (Vite)"]
        UI[Upload Interface]
        PROC[Processing Status]
        DASH[Analytics Dashboard]
    end
    
    subgraph Backend["FastAPI Backend"]
        API[REST API Endpoints]
        JOB[Job Manager]
        PIPE[Analytics Pipeline]
    end
    
    subgraph Vision["Computer Vision Modules"]
        VL[Video Loader]
        FE[Frame Extractor]
        PD[Player Detector YOLO11]
        BD[Ball Detector YOLO11]
        PT[Player Tracker]
        BT[Ball Tracker]
        CD[Court Line Detector]
        CH[Court Homography]
    end
    
    subgraph Analytics["Analytics Modules"]
        BA[Ball Trajectory Analyzer]
        BC[Bounce Court Analyzer]
        BS[Ball Speed Analyzer]
        PM[Player Motion Analyzer]
        SF[Shot Feature Extractor]
        SC[Shot Classifier ML]
        TS[Tennis Scoring Engine]
    end
    
    subgraph Rendering["Video Rendering"]
        VR[Video Renderer]
        OV[Overlay Generator]
    end
    
    UI -->|Upload Video| API
    API -->|Create Job| JOB
    JOB -->|Process| PIPE
    PIPE -->|Extract| VL
    VL --> FE
    FE --> PD
    FE --> BD
    PD --> PT
    BD --> BT
    FE --> CD
    CD --> CH
    BT --> BA
    BA --> BC
    BT --> BS
    PT --> PM
    BA --> SF
    SF --> SC
    BC --> TS
    SC --> TS
    PIPE --> VR
    VR --> OV
    OV -->|Annotated Video| JOB
    JOB -->|Status/Results| API
    API -->|Poll Status| PROC
    API -->|Display Results| DASH
```

---

##  Project Structure

```
TrackScore/
├── trackscore.py             # Main CLI entry point
├── backend/
│   └── app/
│       ├── vision/           # Computer vision modules
│       │   ├── video_loader.py
│       │   ├── player_detector.py
│       │   ├── ball_detector.py
│       │   ├── player_tracker.py
│       │   ├── ball_tracker.py
│       │   ├── court_line_detector.py
│       │   ├── court_homography.py
│       │   ├── court_geometry.py
│       │   └── video_renderer.py
│       ├── analytics/        # Analytics modules
│       │   ├── ball_trajectory_analyzer.py
│       │   ├── bounce_court_analyzer.py
│       │   ├── ball_speed_analyzer.py
│       │   └── player_motion_analyzer.py
│       ├── ml/               # Machine learning
│       │   ├── shot_feature_extractor.py
│       │   └── shot_classifier.py
│       ├── scoring/          # Tennis scoring
│       │   └── tennis_scoring.py
│       └── api/              # FastAPI (optional/legacy)
├── tests/                    # Test suite (237 tests)
├── scripts/                  # Utility scripts
├── data/                     # Training/test data
├── models/                   # ML model weights
├── samples/                  # Sample videos
├── outputs/                  # Generated results
└── requirements.txt
```

---

##  Tech Stack

- **Python 3.8+** - Core language
- **OpenCV** - Computer vision operations
- **Ultralytics YOLO11** - Object detection (players, ball)
- **NumPy** - Numerical computations
- **scikit-learn** - Machine learning (shot classification)
- **pytest** - Test framework (237 tests)

---

##  Getting Started

### Prerequisites

- **Python 3.8+**
- **Git**

### Installation

#### 1. Clone Repository
```bash
git clone <repository-url>
cd TrackScore
```

#### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Verify Setup

```bash
# Run smoke tests
python scripts/smoke_test.py

# Run a quick test
python trackscore.py samples/tennis_match.mp4 --max-seconds 3 --no-preview
```

---

##  Usage

### Basic Command

```bash
python trackscore.py samples/tennis_match.mp4
```

This will:
1. Load and analyze the video
2. Show live annotated preview (press 'q' to quit)
3. Generate annotated video: `outputs/final/trackscore_analysis.mp4`
4. Generate analytics: `outputs/final/analytics.json`
5. Generate summary: `outputs/final/summary.json`

### Command Line Options

```bash
# Process only first N seconds (useful for testing)
python trackscore.py video.mp4 --max-seconds 10

# Disable live preview window
python trackscore.py video.mp4 --no-preview

# Specify output location
python trackscore.py video.mp4 --output results/my_analysis.mp4

# Process every Nth frame (faster, lower quality)
python trackscore.py video.mp4 --frame-stride 2

# Set court type for bounce detection
python trackscore.py video.mp4 --court-type doubles
```

### Full Options

```
python trackscore.py VIDEO [OPTIONS]

Arguments:
  VIDEO                    Path to input tennis video (MP4, MOV, AVI)

Options:
  --output PATH            Output video path (default: outputs/final/trackscore_analysis.mp4)
  --max-seconds FLOAT      Process only first N seconds
  --no-preview             Disable live preview window
  --frame-stride INT       Process every Nth frame (default: 1)
  --court-type TYPE        Court type: singles or doubles (default: singles)
  -h, --help               Show help message
```

---

##  Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test Suite
```bash
# API tests
pytest tests/test_api.py

# Integration tests
pytest tests/test_integration.py

# Vision module tests
pytest tests/test_player_detector.py
pytest tests/test_ball_tracker.py

# Analytics tests
pytest tests/test_bounce_court_analyzer.py
pytest tests/test_ball_speed_analyzer.py

# ML tests
pytest tests/test_shot_classifier.py

# Scoring tests
pytest tests/test_tennis_scoring.py

# Renderer tests
pytest tests/test_video_renderer.py
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Coverage
```bash
pytest --cov=backend
```

### Frontend Tests
```bash
# Note: Frontend has been deprecated in favor of CLI-first approach
# Legacy FastAPI tests remain for reference
pytest tests/test_api.py tests/test_integration.py
```

**Test Coverage:**
- 237 total tests
- API integration tests (29 tests)
- End-to-end workflow tests (9 tests)
- Computer vision module tests (90+ tests)
- Analytics module tests (50+ tests)
- ML pipeline tests (15+ tests)
- Video renderer tests (22 tests)
- Scoring engine tests (18 tests)

---

##  ML Pipeline Explanation

### Shot Classification Workflow

1. **Feature Extraction**
   - Rally segmentation from ball trajectory data
   - Shot segmentation within rallies
   - Feature computation per shot:
     - Ball landing position (court coordinates)
     - Shot angle and direction
     - Ball speed at impact
     - Distance traveled
     - Time between shots
     - Trajectory curvature

2. **Training**
   - Multi-model ensemble approach:
     - Random Forest Classifier
     - Gradient Boosting Classifier
     - Support Vector Machine (SVM)
   - Majority voting for predictions
   - Cross-validation for model evaluation
   - Model serialization with joblib

3. **Prediction**
   - Load trained models from `models/shot/`
   - Extract features from new match data
   - Ensemble prediction across models
   - Shot type classification (forehand, backhand, serve, volley, etc.)

** Training Data Notice:**
- Demo training data in `data/processed/demo_shot_dataset.csv` is **synthetic**
- Generated for demonstration purposes only
- Real-world accuracy requires labeled match footage
- Current models should not be used for production analysis

### Model Training Script
```bash
python scripts/train_shot_classifier.py
```

---

##  Computer Vision Pipeline Explanation

### 1. Video Loading & Frame Extraction
- Load video metadata (resolution, FPS, duration)
- Extract frames at configurable intervals
- Validate video codec and format

### 2. Frame Preprocessing
- Resize to YOLO input dimensions (640x640)
- Letterbox padding to maintain aspect ratio
- Brightness and quality validation
- Normalization for model input

### 3. Player Detection & Tracking
- **Detection**: YOLO11 person detection on each frame
- **Filtering**: Score-based filtering and court-region validation
- **Tracking**: Stable ID assignment using spatial proximity
- **Smoothing**: Position smoothing across frames

### 4. Ball Detection & Tracking
- **Detection**: YOLO11 sports ball detection
- **Validation**: Color-based validation (yellow tennis ball)
- **Tracking**: Kalman filter for trajectory prediction
- **Gap Filling**: Predicted positions during occlusion/miss

### 5. Court Analysis
- **Line Detection**: Canny edge detection + Hough line transform
- **Classification**: Horizontal/vertical line identification
- **Homography**: 2D perspective transform calibration
- **Geometry**: Real-world court coordinate mapping

### 6. Bounce Detection
- **Trajectory Analysis**: Vertical velocity reversal detection
- **Spatial Validation**: Bounce position mapped to court coordinates
- **IN/OUT Classification**: Boundary distance comparison with tolerance
- **Confidence Scoring**: Based on tracking quality and geometry

### 7. Speed Estimation
- **Court-Plane Speed**: 2D distance / time calculations
- **Calibration Required**: Uses homography for pixel-to-meter conversion
- **Limitations**: 
  - Not true 3D speed (only court-plane projection)
  - Accuracy depends on camera angle and calibration
  - Movement perpendicular to court plane is not captured

### 8. Video Rendering
- **Frame-by-frame Processing**: Read source video frames
- **Overlay Generation**: 
  - TrackScore branding with timestamp
  - Live scoreboard (Player A/B, points/games/sets)
  - Player bounding boxes and speed
  - Ball markers (green=detected, orange=predicted)
  - Ball trajectory tail (30 frames)
  - Event overlays (BOUNCE, IN, OUT, SHOT, RALLY)
- **Encoding**: MP4 output with mp4v codec

---

##  Limitations

### Technical Constraints
1. **Monocular Video**: Single-camera footage limits 3D reconstruction accuracy
2. **Speed Estimation**: Court-plane projections only; not true 3D velocities
3. **Calibration Dependency**: Accuracy requires good court line visibility and successful homography
4. **Lighting Conditions**: Performance degrades in poor lighting or shadows
5. **Camera Angle**: Best results with elevated, centered camera positions
6. **Occlusion Handling**: Ball tracking fails during heavy occlusion periods

### Model Limitations
1. **YOLO Detection**: Dependent on model training data; may miss small/distant balls
2. **Shot Classification**: Requires labeled training data; demo models are not production-ready
3. **Scoring Accuracy**: Assumes perfect rally detection; may miss or miscount points
4. **Bounce Detection**: Sensitive to frame rate and tracking quality

### System Limitations
1. **Processing Time**: Real-time processing not supported; batch processing only
2. **File Size**: 100MB upload limit (configurable)
3. **Memory Usage**: Full video loaded into memory during processing
4. **No GPU Acceleration**: CPU-only processing (GPU support possible with PyTorch CUDA)

---

##  Future Improvements

### Computer Vision
- [ ] Multi-camera support for 3D reconstruction
- [ ] Deep learning-based ball tracking (TrackNet, etc.)
- [ ] Player pose estimation for biomechanics analysis
- [ ] Automatic camera calibration without court lines
- [ ] Real-time processing with streaming support

### Machine Learning
- [ ] Large-scale shot classification training dataset
- [ ] Player identification and recognition
- [ ] Stroke quality assessment (spin, placement, power)
- [ ] Match outcome prediction
- [ ] Injury risk detection from movement patterns

### Analytics
- [ ] Heatmaps for player positioning and shot placement
- [ ] Rally pattern analysis and strategy insights
- [ ] Comparative player performance metrics
- [ ] Historical match database and trends
- [ ] Export to standard tennis analytics formats

### System
- [ ] GPU acceleration for faster processing
- [ ] WebSocket for real-time status updates
- [ ] Video streaming support (HLS/DASH)
- [ ] Mobile app for on-court recording
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] User authentication and match history

### User Experience
- [ ] Interactive video player with frame-by-frame navigation
- [ ] Manual correction tools for detections
- [ ] Customizable overlay themes
- [ ] PDF/CSV report generation
- [ ] Social sharing features

---

##  Performance Notes

- **Processing Speed**: ~1-3 FPS on CPU (depends on video resolution)
- **Accuracy**: Varies with video quality; best with HD+ footage from elevated angles
- **Memory Usage**: ~2-4GB for typical 5-minute match video
- **Test Coverage**: 237 tests covering all major components

---

##  License

This project is for educational and portfolio purposes. Not licensed for commercial use.

---

##  Author

Portfolio project demonstrating full-stack development, computer vision, and machine learning integration.

---

##  Acknowledgments

- **Ultralytics YOLO11** for object detection models
- **OpenCV** for computer vision foundations
- **FastAPI** for modern Python web framework
- **React** for frontend UI
- Tennis community for inspiration

---

** Final Disclaimer**: TrackScore is a research and portfolio project. It is not intended for professional match officiating, certified line calling, or any use where accuracy is critical. All analytics are estimates and should be validated with proper equipment and methodology for any serious application.
