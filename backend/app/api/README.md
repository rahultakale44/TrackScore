# TrackScore FastAPI Backend

FastAPI backend for TrackScore tennis video analysis system.

## Features

- **Video Upload**: Upload MP4, MOV, or AVI video files (max 100 MB)
- **Asynchronous Processing**: Jobs run in the background without blocking
- **Status Tracking**: Real-time progress monitoring
- **Result Retrieval**: Structured analysis summaries and full metadata
- **CORS Support**: Cross-origin requests enabled
- **Safe Path Handling**: Protection against path traversal attacks
- **Graceful Error Handling**: Clear error messages with proper HTTP status codes

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

### Development Mode

```bash
uvicorn backend.app.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Production Mode

```bash
uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, visit:

- **Interactive API docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI schema**: http://localhost:8000/openapi.json

## API Endpoints

### Health Check

**GET** `/api/health`

Check if the service is running.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### Video Upload

**POST** `/api/videos/upload`

Upload a video file for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Video file (MP4, MOV, or AVI, max 100 MB)

**Response (201 Created):**
```json
{
  "video_id": "unique-video-id",
  "filename": "original-filename.mp4",
  "size_bytes": 1234567,
  "upload_path": "uploads/unique-video-id.mp4"
}
```

**Errors:**
- `400 Bad Request`: Invalid file extension or missing file
- `413 Content Too Large`: File exceeds 100 MB limit

---

### Start Analysis

**POST** `/api/analysis/start/{video_id}`

Start analyzing an uploaded video.

**Request Body (optional):**
```json
{
  "max_frames": 10
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "unique-job-id",
  "video_id": "video-id",
  "status": "queued"
}
```

**Errors:**
- `404 Not Found`: Video ID not found

---

### Check Job Status

**GET** `/api/analysis/status/{job_id}`

Get the current status of an analysis job.

**Response (200 OK):**
```json
{
  "job_id": "job-id",
  "video_id": "video-id",
  "status": "processing",
  "progress_percentage": 45.0,
  "message": "Processing video frames",
  "error": null
}
```

**Status Values:**
- `queued`: Job is waiting to be processed
- `processing`: Job is currently being processed
- `completed`: Job finished successfully
- `failed`: Job failed with an error

**Errors:**
- `404 Not Found`: Job ID not found

---

### Get Analysis Result

**GET** `/api/analysis/result/{job_id}`

Get structured analysis summary.

**Response (200 OK):**
```json
{
  "job_id": "job-id",
  "video_id": "video-id",
  "status": "completed",
  "summary": {
    "video": {
      "filename": "test.mp4",
      "fps": 30.0,
      "frame_count": 90,
      "duration_seconds": 3.0,
      "resolution": "1920x1080"
    },
    "processing": {
      "frames_processed": 3,
      "status": "success"
    },
    "detections": {
      "players": {
        "count": 2,
        "total_detections": 6
      },
      "ball": {
        "count": 1,
        "total_detections": 3
      }
    }
  },
  "warnings": []
}
```

**Errors:**
- `400 Bad Request`: Analysis not completed yet
- `404 Not Found`: Job ID not found

---

### Get Full Metadata

**GET** `/api/analysis/metadata/{job_id}`

Get complete analytics JSON with all detection data.

**Response (200 OK):**
```json
{
  "job_id": "job-id",
  "video_id": "video-id",
  "status": "completed",
  "metadata": {
    "job_id": "job-id",
    "video_id": "video-id",
    "video_path": "uploads/video.mp4",
    "output_dir": "outputs/jobs/job-id",
    "pipeline_result": {
      "video_metadata": {...},
      "frame_count": 3,
      "frames": [...],
      "detections": {...}
    }
  },
  "warnings": []
}
```

**Errors:**
- `400 Bad Request`: Analysis not completed yet
- `404 Not Found`: Job ID not found

---

### Download Processed Video

**GET** `/api/analysis/video/{job_id}`

Download the processed/annotated video file.

**Response (200 OK):**
- Content-Type: `video/mp4`
- Body: Video file

**Note:** Currently returns the original video. Full annotation will be added in a future commit.

**Errors:**
- `400 Bad Request`: Analysis not completed yet
- `404 Not Found`: Job or video file not found

---

## Usage Example

### Using Python `requests`

```python
import requests
import time

# Upload video
with open("tennis_match.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/videos/upload",
        files={"file": ("tennis_match.mp4", f, "video/mp4")}
    )
    video_id = response.json()["video_id"]

# Start analysis
response = requests.post(
    f"http://localhost:8000/api/analysis/start/{video_id}",
    json={"max_frames": 10}
)
job_id = response.json()["job_id"]

# Poll status
while True:
    response = requests.get(f"http://localhost:8000/api/analysis/status/{job_id}")
    status = response.json()["status"]
    
    if status == "completed":
        break
    elif status == "failed":
        print("Analysis failed!")
        break
    
    time.sleep(1)

# Get result
response = requests.get(f"http://localhost:8000/api/analysis/result/{job_id}")
result = response.json()
print(result["summary"])
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/api/health

# Upload video
curl -X POST http://localhost:8000/api/videos/upload \
  -F "file=@tennis_match.mp4"

# Start analysis
curl -X POST http://localhost:8000/api/analysis/start/{video_id} \
  -H "Content-Type: application/json" \
  -d '{"max_frames": 10}'

# Check status
curl http://localhost:8000/api/analysis/status/{job_id}

# Get result
curl http://localhost:8000/api/analysis/result/{job_id}
```

## Architecture

### Components

- **`main.py`**: FastAPI application factory and middleware configuration
- **`models.py`**: Pydantic models for request/response validation
- **`job_manager.py`**: In-memory job state manager
- **`routes/`**: API endpoint implementations
  - `health.py`: Health check endpoint
  - `videos.py`: Video upload endpoint
  - `analysis.py`: Analysis job endpoints

### Job Processing Flow

1. **Upload**: Client uploads video → Server saves to `uploads/` directory → Returns `video_id`
2. **Start**: Client requests analysis → Server creates job → Returns `job_id` → Processing starts asynchronously
3. **Process**: Background task runs `VideoAnalyticsPipeline` → Updates job status and progress
4. **Complete**: Processing finishes → Results stored in job manager → Client can retrieve results

### Storage

- **Videos**: `uploads/{video_id}.{ext}`
- **Job outputs**: `outputs/jobs/{job_id}/`
- **Job state**: In-memory (lost on server restart)

### Error Handling

- Pipeline failures are captured as warnings when possible
- Partial results are returned with warnings instead of complete failure
- Clear error messages with appropriate HTTP status codes
- Safe path handling prevents directory traversal attacks

## Testing

Run the API tests:

```bash
python -m pytest tests/test_api.py -v
```

Run a demo script:

```bash
# Make sure server is running first
python scripts/demo_api.py
```

## Configuration

### File Upload Limits

Edit `backend/app/api/routes/videos.py`:

```python
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
```

### Supported Extensions

Edit `backend/app/api/routes/videos.py`:

```python
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi"}
```

### CORS Settings

Edit `backend/app/api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Future Enhancements

- Persistent job storage (database)
- Video annotation overlay
- Real-time WebSocket status updates
- Rate limiting and authentication
- Job cleanup and retention policies
- Multi-video batch processing
- Result export formats (JSON, CSV, PDF)

## Troubleshooting

### Server won't start

- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: requires Python 3.10+

### Video upload fails

- Check file size (max 100 MB)
- Verify file extension (MP4, MOV, or AVI only)
- Ensure `uploads/` directory is writable

### Analysis job stays in "queued" status

- Check server logs for errors
- Verify video file exists in `uploads/` directory
- Ensure pipeline dependencies are installed (OpenCV, YOLO, etc.)

### "Module not found" errors

```bash
pip install --upgrade -r requirements.txt
```

## License

Part of the TrackScore tennis analytics system.
