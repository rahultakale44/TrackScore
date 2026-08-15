#!/usr/bin/env python3
"""
Verify court calibration by drawing keypoints and testing transformation.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.vision.video_loader import VideoLoader
from backend.app.vision.court_calibrator import CourtCalibrator

def main():
    video_path = "samples/tennis_test2.mp4"
    calib_path = "data/court_calibration.json"
    
    # Load video
    loader = VideoLoader(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame")
        return 1
    
    # Load calibration
    calibrator = CourtCalibrator()
    calibrator.load_calibration(calib_path)
    
    print("Court Calibration Verification")
    print("=" * 70)
    
    # Print keypoints
    kp = calibrator.keypoints
    print("\nImage Keypoints:")
    print(f"  Far baseline left:  {kp.far_baseline_left}")
    print(f"  Far baseline right: {kp.far_baseline_right}")
    print(f"  Near baseline left:  {kp.near_baseline_left}")
    print(f"  Near baseline right: {kp.near_baseline_right}")
    print(f"  Far service left:  {kp.far_service_left}")
    print(f"  Far service right: {kp.far_service_right}")
    print(f"  Near service left:  {kp.near_service_left}")
    print(f"  Near service right: {kp.near_service_right}")
    print(f"  Net left:  {kp.net_left}")
    print(f"  Net right: {kp.net_right}")
    
    # Test transformation
    print("\nTest Transformations:")
    test_points = {
        "Far baseline left": kp.far_baseline_left,
        "Far baseline right": kp.far_baseline_right,
        "Near baseline left": kp.near_baseline_left,
        "Near baseline right": kp.near_baseline_right,
        "Net center": ((kp.net_left[0] + kp.net_right[0])/2, (kp.net_left[1] + kp.net_right[1])/2),
    }
    
    for name, img_pt in test_points.items():
        court_pt = calibrator.image_to_court_coordinates(img_pt)
        if court_pt:
            print(f"  {name}: image {img_pt} → court ({court_pt[0]:.2f}, {court_pt[1]:.2f}) m")
        else:
            print(f"  {name}: transformation failed")
    
    # Draw overlay
    overlay = calibrator.draw_court_overlay(frame)
    
    # Save
    output_path = "outputs/final/calibration_verification.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"\nOverlay saved to: {output_path}")
    
    # Show expected court bounds
    print("\nExpected Court Bounds (singles):")
    print(f"  X: -{calibrator.court_model.COURT_WIDTH/2:.2f} to +{calibrator.court_model.COURT_WIDTH/2:.2f} meters")
    print(f"  Y: 0.0 to {calibrator.court_model.COURT_LENGTH:.2f} meters")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
