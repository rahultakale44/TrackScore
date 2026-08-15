#!/usr/bin/env python3
"""
Smoke test for TrackScore end-to-end workflow.

This script tests the basic API functionality without starting a full server.
For a complete test, start the FastAPI server and run integration tests.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import Config


def test_directories():
    """Test that all required directories exist."""
    print("✓ Checking required directories...")
    
    assert Config.UPLOAD_DIR.exists(), f"Upload directory missing: {Config.UPLOAD_DIR}"
    assert Config.OUTPUT_DIR.exists(), f"Output directory missing: {Config.OUTPUT_DIR}"
    assert (Config.OUTPUT_DIR / "jobs").exists(), f"Jobs directory missing"
    
    print(f"  - Upload directory: {Config.UPLOAD_DIR}")
    print(f"  - Output directory: {Config.OUTPUT_DIR}")
    print(f"  - Jobs directory: {Config.OUTPUT_DIR / 'jobs'}")


def test_configuration():
    """Test that configuration is loaded correctly."""
    print("\n✓ Checking configuration...")
    
    print(f"  - API Host: {Config.API_HOST}")
    print(f"  - API Port: {Config.API_PORT}")
    print(f"  - Max upload size: {Config.MAX_UPLOAD_SIZE / (1024 * 1024):.0f} MB")
    print(f"  - Allowed extensions: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}")
    print(f"  - Log level: {Config.LOG_LEVEL}")


def test_imports():
    """Test that all critical modules can be imported."""
    print("\n✓ Checking module imports...")
    
    try:
        from backend.app.api.main import create_app
        print("  - FastAPI app: OK")
    except Exception as e:
        print(f"  - FastAPI app: FAILED ({e})")
        return False
    
    try:
        from backend.app.api.job_manager import JobManager
        print("  - Job Manager: OK")
    except Exception as e:
        print(f"  - Job Manager: FAILED ({e})")
        return False
    
    try:
        from backend.app.vision.video_pipeline import VideoAnalyticsPipeline
        print("  - Video Pipeline: OK")
    except Exception as e:
        print(f"  - Video Pipeline: FAILED ({e})")
        return False
    
    try:
        from backend.app.vision.video_renderer import VideoRenderer
        print("  - Video Renderer: OK")
    except Exception as e:
        print(f"  - Video Renderer: FAILED ({e})")
        return False
    
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("TrackScore Smoke Test")
    print("=" * 60)
    
    try:
        test_directories()
        test_configuration()
        if not test_imports():
            print("\n✕ Some imports failed!")
            return 1
        
        print("\n" + "=" * 60)
        print("✓ All smoke tests passed!")
        print("=" * 60)
        print("\nTo test the full workflow:")
        print("  1. Start backend: python -m uvicorn backend.app.api.main:app --reload")
        print("  2. Start frontend: cd frontend && npm run dev")
        print("  3. Run integration tests: pytest tests/test_integration.py")
        
        return 0
    
    except Exception as e:
        print(f"\n✕ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
