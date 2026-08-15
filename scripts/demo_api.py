"""
Demo script to test the FastAPI video analysis endpoints.

Run with: python scripts/demo_api.py
"""

import time
from pathlib import Path

import requests

# API base URL
API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{API_BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_video_upload():
    """Test video upload."""
    print("\n=== Testing Video Upload ===")
    
    # Create a small fake video file for testing
    test_file_path = Path("test_demo_video.mp4")
    test_file_path.write_bytes(b"fake video content for demo")
    
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": ("demo_video.mp4", f, "video/mp4")}
            response = requests.post(f"{API_BASE_URL}/api/videos/upload", files=files)
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Video ID: {data.get('video_id')}")
        print(f"Filename: {data.get('filename')}")
        print(f"Size: {data.get('size_bytes')} bytes")
        
        return data.get("video_id") if response.status_code == 201 else None
    finally:
        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()


def test_start_analysis(video_id):
    """Test starting an analysis job."""
    print(f"\n=== Starting Analysis for Video {video_id} ===")
    
    response = requests.post(
        f"{API_BASE_URL}/api/analysis/start/{video_id}",
        json={"max_frames": 1}
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Job ID: {data.get('job_id')}")
    print(f"Status: {data.get('status')}")
    
    return data.get("job_id") if response.status_code == 202 else None


def test_check_status(job_id):
    """Test checking job status."""
    print(f"\n=== Checking Status for Job {job_id} ===")
    
    # Poll status until completion or failure
    max_attempts = 30
    for attempt in range(max_attempts):
        response = requests.get(f"{API_BASE_URL}/api/analysis/status/{job_id}")
        
        if response.status_code != 200:
            print(f"Error checking status: {response.status_code}")
            return False
        
        data = response.json()
        status = data.get("status")
        progress = data.get("progress_percentage")
        message = data.get("message")
        
        print(f"Attempt {attempt + 1}: Status={status}, Progress={progress}%, Message={message}")
        
        if status == "completed":
            return True
        elif status == "failed":
            print(f"Job failed: {data.get('error')}")
            return False
        
        time.sleep(1)
    
    print("Timeout waiting for job completion")
    return False


def test_get_result(job_id):
    """Test getting analysis result."""
    print(f"\n=== Getting Result for Job {job_id} ===")
    
    response = requests.get(f"{API_BASE_URL}/api/analysis/result/{job_id}")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        return False
    
    data = response.json()
    print(f"Status: {data.get('status')}")
    print(f"Summary: {data.get('summary')}")
    print(f"Warnings: {data.get('warnings')}")
    
    return True


def test_get_metadata(job_id):
    """Test getting full analysis metadata."""
    print(f"\n=== Getting Metadata for Job {job_id} ===")
    
    response = requests.get(f"{API_BASE_URL}/api/analysis/metadata/{job_id}")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return False
    
    data = response.json()
    print(f"Status: {data.get('status')}")
    print(f"Metadata keys: {list(data.get('metadata', {}).keys())}")
    
    return True


def main():
    """Run all API tests."""
    print("=" * 60)
    print("FastAPI Video Analysis Demo")
    print("=" * 60)
    print("\nMake sure the API server is running:")
    print("  uvicorn backend.app.api.main:app --reload")
    print("=" * 60)
    
    try:
        # Test health check
        if not test_health_check():
            print("\n❌ Health check failed!")
            return
        
        # Test video upload
        video_id = test_video_upload()
        if not video_id:
            print("\n❌ Video upload failed!")
            return
        
        # Test starting analysis
        job_id = test_start_analysis(video_id)
        if not job_id:
            print("\n❌ Starting analysis failed!")
            return
        
        # Test checking status
        if not test_check_status(job_id):
            print("\n❌ Job did not complete successfully!")
            # Still continue to show what endpoints are available
        
        # Test getting result
        test_get_result(job_id)
        
        # Test getting metadata
        test_get_metadata(job_id)
        
        print("\n" + "=" * 60)
        print("✅ Demo completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server!")
        print("Make sure the server is running:")
        print("  uvicorn backend.app.api.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
