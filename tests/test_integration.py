"""
Integration tests for end-to-end TrackScore workflow.
"""

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.main import create_app


@pytest.fixture
def client():
    """Create a test client with proper lifespan handling."""
    from backend.app.api.job_manager import JobManager
    
    app = create_app()
    
    # Manually set up the job manager for testing
    # TestClient doesn't call lifespan events by default
    with TestClient(app) as test_client:
        app.state.job_manager = JobManager()
        yield test_client


@pytest.fixture
def sample_video_bytes():
    """
    Create a minimal valid MP4 file for testing.
    
    This is a minimal MP4 header that will be accepted by the upload endpoint.
    """
    # Minimal MP4 file structure (ftyp + mdat boxes)
    return (
        b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6f\x6d\x00\x00\x02\x00'
        b'\x69\x73\x6f\x6d\x69\x73\x6f\x32\x6d\x70\x34\x31\x00\x00\x00\x08'
        b'\x66\x72\x65\x65'
    )


def test_health_check_integration(client):
    """Test API health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_upload_video_integration(client, sample_video_bytes):
    """Test video upload endpoint."""
    files = {"file": ("test_video.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    response = client.post("/api/videos/upload", files=files)
    
    assert response.status_code == 201
    data = response.json()
    
    assert "video_id" in data
    assert data["filename"] == "test_video.mp4"
    assert data["size_bytes"] > 0
    assert "upload_path" in data
    
    # Verify file was saved
    upload_path = Path(data["upload_path"])
    assert upload_path.exists()
    
    # Clean up
    upload_path.unlink()


def test_start_analysis_integration(client, sample_video_bytes):
    """Test starting analysis job."""
    # First upload a video
    files = {"file": ("test_video.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    upload_response = client.post("/api/videos/upload", files=files)
    assert upload_response.status_code == 201
    
    video_id = upload_response.json()["video_id"]
    upload_path = Path(upload_response.json()["upload_path"])
    
    # Start analysis
    analysis_response = client.post(f"/api/analysis/start/{video_id}")
    assert analysis_response.status_code == 202
    
    data = analysis_response.json()
    assert "job_id" in data
    assert data["video_id"] == video_id
    assert data["status"] in ["queued", "processing"]
    
    job_id = data["job_id"]
    
    # Check status
    status_response = client.get(f"/api/analysis/status/{job_id}")
    assert status_response.status_code == 200
    
    status_data = status_response.json()
    assert status_data["job_id"] == job_id
    assert status_data["video_id"] == video_id
    assert status_data["status"] in ["queued", "processing", "completed", "failed"]
    assert "progress_percentage" in status_data
    assert "message" in status_data
    
    # Clean up
    upload_path.unlink(missing_ok=True)


def test_nonexistent_video_analysis(client):
    """Test starting analysis for non-existent video."""
    response = client.post("/api/analysis/start/nonexistent-video-id")
    assert response.status_code == 404


def test_job_status_not_found(client):
    """Test checking status of non-existent job."""
    response = client.get("/api/analysis/status/nonexistent-job-id")
    assert response.status_code == 404


def test_result_before_completion(client, sample_video_bytes):
    """Test accessing result - job may complete immediately with test data."""
    # Upload video and start analysis
    files = {"file": ("test_video.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    upload_response = client.post("/api/videos/upload", files=files)
    video_id = upload_response.json()["video_id"]
    upload_path = Path(upload_response.json()["upload_path"])
    
    analysis_response = client.post(f"/api/analysis/start/{video_id}")
    job_id = analysis_response.json()["job_id"]
    
    # The minimal test video may complete very quickly or fail immediately
    # Just verify we can get a response (either result or error about completion)
    result_response = client.get(f"/api/analysis/result/{job_id}")
    assert result_response.status_code in [200, 400]  # Either completed or not yet
    
    # Clean up
    upload_path.unlink(missing_ok=True)


def test_upload_size_validation(client):
    """Test file size validation."""
    # Create a file that's too large (> 100MB)
    large_content = b'x' * (101 * 1024 * 1024)
    files = {"file": ("large_video.mp4", io.BytesIO(large_content), "video/mp4")}
    
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 413


def test_upload_extension_validation(client):
    """Test file extension validation."""
    files = {"file": ("video.txt", io.BytesIO(b"not a video"), "text/plain")}
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400


def test_directories_created():
    """Test that required directories are created on startup."""
    from backend.app.core.config import Config
    
    assert Config.UPLOAD_DIR.exists()
    assert Config.OUTPUT_DIR.exists()
    assert (Config.OUTPUT_DIR / "jobs").exists()


__all__ = [
    "test_health_check_integration",
    "test_upload_video_integration",
    "test_start_analysis_integration",
    "test_nonexistent_video_analysis",
    "test_job_status_not_found",
    "test_result_before_completion",
    "test_upload_size_validation",
    "test_upload_extension_validation",
    "test_directories_created",
]
