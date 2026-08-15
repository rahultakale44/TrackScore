#!/usr/bin/env python3
"""
Verify TrackScore setup and dependencies.

Checks that all required Python packages and system components are available.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✕ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check critical dependencies."""
    print("\nChecking Python dependencies...")
    
    required = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics YOLO"),
    ]
    
    all_ok = True
    for module_name, display_name in required:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✕ {display_name} not found")
            all_ok = False
    
    return all_ok


def check_yolo_model():
    """Check YOLO model file."""
    print("\nChecking YOLO model...")
    
    model_path = Path("yolo11n.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ YOLO model found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ⚠ YOLO model not found (will auto-download on first use)")
        return True  # Not critical, will download


def check_sample_video():
    """Check sample video."""
    print("\nChecking sample video...")
    
    sample_path = Path("samples/tennis_match.mp4")
    if sample_path.exists():
        size_mb = sample_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Sample video found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ⚠ Sample video not found")
        return True  # Not critical


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("TrackScore Setup Verification")
    print("=" * 60 + "\n")
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_yolo_model(),
        check_sample_video(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks[:2]):  # Only first two are critical
        print("✓ Setup verification passed!")
        print("=" * 60)
        print("\nReady to run TrackScore.")
        print("\nNext steps:")
        print("  1. Install frontend dependencies: cd frontend && npm install")
        print("  2. Start backend: python -m uvicorn backend.app.api.main:app --reload")
        print("  3. Start frontend: cd frontend && npm run dev")
        return 0
    else:
        print("✕ Setup verification failed!")
        print("=" * 60)
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
