"""
Tests for FastAPI video analysis API.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.api.job_manager import JobManager
from backend.app.api.main import create_app
from backend.app.api.models import JobStatus


@pytest.fixture
def app():
    """Create FastAPI test application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_job_manager():
    """Create a mock job manager."""
    return MagicMock(spec=JobManager)


@pytest.fixture
def temp_upload_dir(tmp_path, monkeypatch):
    """Create temporary upload directory."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    
    # Patch the UPLOAD_DIR in the videos module
    from backend.app.api.routes import videos
    monkeypatch.setattr(videos, "UPLOAD_DIR", upload_dir)
    
    from backend.app.api.routes import analysis
    monkeypatch.setattr(analysis, "UPLOAD_DIR", upload_dir)
    
    return upload_dir


# ============================================================
# HEALTH ENDPOINT TESTS
# ============================================================


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


# ============================================================
# VIDEO UPLOAD TESTS
# ============================================================


def test_upload_video_success(client, temp_upload_dir):
    """Test successful video upload."""
    # Create fake video content
    video_content = b"fake video content"
    video_file = io.BytesIO(video_content)
    
    response = client.post(
        "/api/videos/upload",
        files={"file": ("test_video.mp4", video_file, "video/mp4")},
    )
    
    assert response.status_code == 201
    data = response.json()
    
    assert "video_id" in data
    assert data["filename"] == "test_video.mp4"
    assert data["size_bytes"] == len(video_content)
    assert "upload_path" in data
    
    # Verify file was saved
    video_id = data["video_id"]
    upload_path = temp_upload_dir / f"{video_id}.mp4"
    assert upload_path.exists()
    assert upload_path.read_bytes() == video_content


def test_upload_video_unsupported_extension(client, temp_upload_dir):
    """Test upload with unsupported file extension."""
    video_file = io.BytesIO(b"fake content")
    
    response = client.post(
        "/api/videos/upload",
        files={"file": ("test_video.txt", video_file, "text/plain")},
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file extension" in data["detail"]


def test_upload_video_too_large(client, temp_upload_dir):
    """Test upload with file size exceeding limit."""
    # Create content larger than 100 MB
    large_content = b"x" * (101 * 1024 * 1024)
    video_file = io.BytesIO(large_content)
    
    response = client.post(
        "/api/videos/upload",
        files={"file": ("large_video.mp4", video_file, "video/mp4")},
    )
    
    assert response.status_code == 413
    data = response.json()
    assert "exceeds maximum allowed size" in data["detail"]


def test_upload_video_mov_format(client, temp_upload_dir):
    """Test upload with MOV format."""
    video_content = b"fake mov content"
    video_file = io.BytesIO(video_content)
    
    response = client.post(
        "/api/videos/upload",
        files={"file": ("test_video.mov", video_file, "video/quicktime")},
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test_video.mov"


def test_upload_video_avi_format(client, temp_upload_dir):
    """Test upload with AVI format."""
    video_content = b"fake avi content"
    video_file = io.BytesIO(video_content)
    
    response = client.post(
        "/api/videos/upload",
        files={"file": ("test_video.avi", video_file, "video/x-msvideo")},
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test_video.avi"


def test_upload_sanitizes_filename(client, temp_upload_dir):
    """Test that dangerous filenames are sanitized."""
    video_content = b"fake content"
    video_file = io.BytesIO(video_content)
    
    # Filename with path traversal attempt
    dangerous_name = "../../../etc/passwd.mp4"
    
    response = client.post(
        "/api/videos/upload",
        files={"file": (dangerous_name, video_file, "video/mp4")},
    )
    
    assert response.status_code == 201
    data = response.json()
    
    # Original filename should be preserved in response
    assert data["filename"] == dangerous_name
    
    # But the actual saved file should be sanitized
    video_id = data["video_id"]
    upload_path = temp_upload_dir / f"{video_id}.mp4"
    assert upload_path.exists()
    
    # Should not create files outside upload directory
    assert not (temp_upload_dir.parent.parent.parent / "etc" / "passwd.mp4").exists()


# ============================================================
# ANALYSIS START TESTS
# ============================================================


def test_start_analysis_success(client, temp_upload_dir, app):
    """Test starting analysis job."""
    # Create a fake uploaded video
    video_id = "test-video-123"
    video_path = temp_upload_dir / f"{video_id}.mp4"
    video_path.write_bytes(b"fake video")
    
    # Mock the job manager
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.create_job.return_value = "job-123"
    app.state.job_manager = mock_job_manager
    
    response = client.post(f"/api/analysis/start/{video_id}")
    
    assert response.status_code == 202
    data = response.json()
    
    assert data["job_id"] == "job-123"
    assert data["video_id"] == video_id
    assert data["status"] == JobStatus.QUEUED
    
    # Verify job was created
    mock_job_manager.create_job.assert_called_once()
    mock_job_manager.start_job_async.assert_called_once_with("job-123")


def test_start_analysis_with_max_frames(client, temp_upload_dir, app):
    """Test starting analysis with max_frames parameter."""
    video_id = "test-video-456"
    video_path = temp_upload_dir / f"{video_id}.mp4"
    video_path.write_bytes(b"fake video")
    
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.create_job.return_value = "job-456"
    app.state.job_manager = mock_job_manager
    
    response = client.post(
        f"/api/analysis/start/{video_id}",
        json={"max_frames": 10},
    )
    
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == "job-456"
    
    # Verify max_frames was passed
    call_args = mock_job_manager.create_job.call_args
    assert call_args[1]["max_frames"] == 10


def test_start_analysis_video_not_found(client, temp_upload_dir, app):
    """Test starting analysis for non-existent video."""
    mock_job_manager = MagicMock(spec=JobManager)
    app.state.job_manager = mock_job_manager
    
    response = client.post("/api/analysis/start/nonexistent-video")
    
    assert response.status_code == 404
    data = response.json()
    assert "Video not found" in data["detail"]


# ============================================================
# ANALYSIS STATUS TESTS
# ============================================================


def test_get_analysis_status_queued(client, app):
    """Test getting status of queued job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.QUEUED,
        "progress_percentage": 0.0,
        "message": "Job queued for processing",
        "error": None,
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/status/job-123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["job_id"] == "job-123"
    assert data["status"] == JobStatus.QUEUED
    assert data["progress_percentage"] == 0.0
    assert data["error"] is None


def test_get_analysis_status_processing(client, app):
    """Test getting status of processing job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.PROCESSING,
        "progress_percentage": 45.0,
        "message": "Processing video frames",
        "error": None,
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/status/job-123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == JobStatus.PROCESSING
    assert data["progress_percentage"] == 45.0
    assert "Processing" in data["message"]


def test_get_analysis_status_completed(client, app):
    """Test getting status of completed job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.COMPLETED,
        "progress_percentage": 100.0,
        "message": "Analysis completed successfully",
        "error": None,
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/status/job-123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == JobStatus.COMPLETED
    assert data["progress_percentage"] == 100.0


def test_get_analysis_status_failed(client, app):
    """Test getting status of failed job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.FAILED,
        "progress_percentage": 0.0,
        "message": "Analysis failed",
        "error": "Video file corrupted",
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/status/job-123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == JobStatus.FAILED
    assert data["error"] == "Video file corrupted"


def test_get_analysis_status_not_found(client, app):
    """Test getting status of non-existent job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = None
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/status/nonexistent-job")
    
    assert response.status_code == 404
    data = response.json()
    assert "Job not found" in data["detail"]


# ============================================================
# ANALYSIS RESULT TESTS
# ============================================================


def test_get_analysis_result_success(client, app):
    """Test getting analysis result."""
    mock_result = {
        "video": {
            "filename": "test.mp4",
            "fps": 30.0,
            "frame_count": 90,
            "duration_seconds": 3.0,
            "resolution": "1920x1080",
        },
        "processing": {
            "frames_processed": 3,
            "status": "success",
        },
        "detections": {
            "players": {
                "count": 2,
                "total_detections": 6,
            },
            "ball": {
                "count": 1,
                "total_detections": 3,
            },
        },
    }
    
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.COMPLETED,
        "result": mock_result,
        "warnings": [],
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/result/job-123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["job_id"] == "job-123"
    assert data["status"] == JobStatus.COMPLETED
    assert data["summary"]["video"]["filename"] == "test.mp4"
    assert data["summary"]["detections"]["players"]["count"] == 2


def test_get_analysis_result_with_warnings(client, app):
    """Test getting analysis result with warnings."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.COMPLETED,
        "result": {"status": "partial"},
        "warnings": ["Pipeline partial failure: some frames skipped"],
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/result/job-123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["warnings"]) == 1
    assert "partial failure" in data["warnings"][0]


def test_get_analysis_result_not_completed(client, app):
    """Test getting result for incomplete job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.PROCESSING,
        "result": None,
        "warnings": [],
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/result/job-123")
    
    assert response.status_code == 400
    data = response.json()
    assert "not completed" in data["detail"].lower()


# ============================================================
# ANALYSIS METADATA TESTS
# ============================================================


def test_get_analysis_metadata_success(client, app):
    """Test getting full analytics metadata."""
    mock_metadata = {
        "job_id": "job-123",
        "video_id": "video-123",
        "pipeline_result": {
            "video_metadata": {"fps": 30.0},
            "frame_count": 3,
            "detections": {"players": [], "ball": []},
        },
    }
    
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.COMPLETED,
        "metadata": mock_metadata,
        "warnings": [],
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/metadata/job-123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["job_id"] == "job-123"
    assert data["status"] == JobStatus.COMPLETED
    assert "pipeline_result" in data["metadata"]


def test_get_analysis_metadata_not_completed(client, app):
    """Test getting metadata for incomplete job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.QUEUED,
        "metadata": None,
        "warnings": [],
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/metadata/job-123")
    
    assert response.status_code == 400
    data = response.json()
    assert "not completed" in data["detail"].lower()


# ============================================================
# ANALYSIS VIDEO TESTS
# ============================================================


def test_get_analysis_video_success(client, temp_upload_dir, app):
    """Test downloading processed video."""
    # Create a fake video file
    video_path = temp_upload_dir / "test-video.mp4"
    video_path.write_bytes(b"fake video content")
    
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.COMPLETED,
        "video_path": str(video_path),
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/video/job-123")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"fake video content"


def test_get_analysis_video_not_completed(client, app):
    """Test downloading video for incomplete job."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.PROCESSING,
        "video_path": "/path/to/video.mp4",
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/video/job-123")
    
    assert response.status_code == 400


def test_get_analysis_video_file_not_found(client, app):
    """Test downloading video when file is missing."""
    mock_job_manager = MagicMock(spec=JobManager)
    mock_job_manager.get_job.return_value = {
        "job_id": "job-123",
        "video_id": "video-123",
        "status": JobStatus.COMPLETED,
        "video_path": "/nonexistent/path/video.mp4",
    }
    app.state.job_manager = mock_job_manager
    
    response = client.get("/api/analysis/video/job-123")
    
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


# ============================================================
# CORS TESTS
# ============================================================


def test_cors_middleware_configured():
    """Test that CORS middleware is configured on the app."""
    # Create a fresh app to test middleware configuration
    from backend.app.api.main import create_app
    test_app = create_app()
    
    # CORS middleware should be present in the middleware stack
    # FastAPI wraps middleware, so we check that it's been added
    assert len(test_app.user_middleware) > 0
    
    # Verify we can make a request (CORS is working)
    from fastapi.testclient import TestClient
    client = TestClient(test_app)
    response = client.get("/api/health")
    assert response.status_code == 200


# ============================================================
# JOB MANAGER TESTS
# ============================================================


def test_job_manager_create_job():
    """Test creating a job."""
    manager = JobManager()
    
    job_id = manager.create_job(
        video_id="video-123",
        video_path="/path/to/video.mp4",
        max_frames=10,
    )
    
    assert job_id is not None
    assert len(job_id) > 0
    
    job = manager.get_job(job_id)
    assert job is not None
    assert job["video_id"] == "video-123"
    assert job["status"] == JobStatus.QUEUED
    assert job["max_frames"] == 10


def test_job_manager_update_status():
    """Test updating job status."""
    manager = JobManager()
    job_id = manager.create_job("video-123", "/path/to/video.mp4")
    
    manager.update_job_status(
        job_id,
        JobStatus.PROCESSING,
        progress_percentage=50.0,
        message="Halfway done",
    )
    
    job = manager.get_job(job_id)
    assert job["status"] == JobStatus.PROCESSING
    assert job["progress_percentage"] == 50.0
    assert job["message"] == "Halfway done"


def test_job_manager_add_warning():
    """Test adding warnings to a job."""
    manager = JobManager()
    job_id = manager.create_job("video-123", "/path/to/video.mp4")
    
    manager.add_job_warning(job_id, "Warning: some frames skipped")
    manager.add_job_warning(job_id, "Warning: low quality detected")
    
    job = manager.get_job(job_id)
    assert len(job["warnings"]) == 2
    assert "skipped" in job["warnings"][0]


def test_job_manager_set_result():
    """Test storing job result."""
    manager = JobManager()
    job_id = manager.create_job("video-123", "/path/to/video.mp4")
    
    result = {"status": "success", "detections": 10}
    manager.set_job_result(job_id, result)
    
    job = manager.get_job(job_id)
    assert job["result"] == result


def test_job_manager_nonexistent_job():
    """Test operations on non-existent job."""
    manager = JobManager()
    
    job = manager.get_job("nonexistent-job")
    assert job is None
    
    # Should not raise errors
    manager.update_job_status("nonexistent-job", JobStatus.COMPLETED)
    manager.add_job_warning("nonexistent-job", "test warning")
    manager.set_job_result("nonexistent-job", {})
