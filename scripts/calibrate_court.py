#!/usr/bin/env python3
"""
Interactive tennis court calibration tool.

Usage:
    python scripts/calibrate_court.py samples/tennis_match.mp4 --frame 50

Click court keypoints in this order:
    1. Far baseline left
    2. Far baseline right
    3. Near baseline left
    4. Near baseline right
    5. Far service line left
    6. Far service line right
    7. Near service line left
    8. Near service line right
    9. Net left
    10. Net right
    11. Far center T (service line center)
    12. Near center T (service line center)

Press 's' to save calibration
Press 'r' to reset points
Press 'q' to quit without saving
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.vision.video_loader import VideoLoader
from backend.app.vision.court_calibrator import CourtCalibrator, CourtKeypoints


class InteractiveCalibrator:
    """Interactive court calibration tool."""
    
    POINT_NAMES = [
        "Far baseline LEFT",
        "Far baseline RIGHT",
        "Near baseline LEFT",
        "Near baseline RIGHT",
        "Far service line LEFT",
        "Far service line RIGHT",
        "Near service line LEFT",
        "Near service line RIGHT",
        "Net LEFT",
        "Net RIGHT",
        "Far center T",
        "Near center T",
    ]
    
    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.points: list = []
        self.calibrator = CourtCalibrator()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < len(self.POINT_NAMES):
                self.points.append((float(x), float(y)))
                self.update_display()
    
    def update_display(self):
        """Redraw display with current points."""
        self.display_frame = self.frame.copy()
        
        # Draw existing points
        for i, pt in enumerate(self.points):
            # Draw circle
            cv2.circle(
                self.display_frame,
                (int(pt[0]), int(pt[1])),
                8,
                (0, 255, 0),
                -1
            )
            
            # Draw label
            label = f"{i+1}"
            cv2.putText(
                self.display_frame,
                label,
                (int(pt[0]) + 12, int(pt[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Draw lines between points if enough collected
        if len(self.points) >= 4:
            # Draw baselines
            cv2.line(self.display_frame,
                    (int(self.points[0][0]), int(self.points[0][1])),
                    (int(self.points[1][0]), int(self.points[1][1])),
                    (0, 255, 255), 2)
            cv2.line(self.display_frame,
                    (int(self.points[2][0]), int(self.points[2][1])),
                    (int(self.points[3][0]), int(self.points[3][1])),
                    (0, 255, 255), 2)
            # Draw sidelines
            cv2.line(self.display_frame,
                    (int(self.points[0][0]), int(self.points[0][1])),
                    (int(self.points[2][0]), int(self.points[2][1])),
                    (0, 255, 255), 2)
            cv2.line(self.display_frame,
                    (int(self.points[1][0]), int(self.points[1][1])),
                    (int(self.points[3][0]), int(self.points[3][1])),
                    (0, 255, 255), 2)
        
        if len(self.points) >= 8:
            # Draw service lines
            cv2.line(self.display_frame,
                    (int(self.points[4][0]), int(self.points[4][1])),
                    (int(self.points[5][0]), int(self.points[5][1])),
                    (0, 255, 255), 2)
            cv2.line(self.display_frame,
                    (int(self.points[6][0]), int(self.points[6][1])),
                    (int(self.points[7][0]), int(self.points[7][1])),
                    (0, 255, 255), 2)
        
        if len(self.points) >= 10:
            # Draw net
            cv2.line(self.display_frame,
                    (int(self.points[8][0]), int(self.points[8][1])),
                    (int(self.points[9][0]), int(self.points[9][1])),
                    (0, 255, 255), 2)
        
        if len(self.points) >= 12:
            # Draw center service line
            cv2.line(self.display_frame,
                    (int(self.points[10][0]), int(self.points[10][1])),
                    (int(self.points[11][0]), int(self.points[11][1])),
                    (0, 255, 255), 2)
        
        # Show instructions
        next_idx = len(self.points)
        if next_idx < len(self.POINT_NAMES):
            instruction = f"Click: {self.POINT_NAMES[next_idx]} ({next_idx+1}/{len(self.POINT_NAMES)})"
        else:
            instruction = "Complete! Press 's' to save, 'r' to reset"
        
        # Draw instruction box
        cv2.rectangle(
            self.display_frame,
            (10, 10),
            (min(len(instruction) * 11 + 20, self.frame.shape[1] - 10), 50),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            self.display_frame,
            instruction,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    
    def reset(self):
        """Clear all points."""
        self.points = []
        self.update_display()
    
    def save_calibration(self, output_path: str) -> bool:
        """Save calibration to file."""
        if len(self.points) != len(self.POINT_NAMES):
            print(f"Error: Need {len(self.POINT_NAMES)} points, only have {len(self.points)}")
            return False
        
        try:
            # Create keypoints object
            keypoints = CourtKeypoints(
                far_baseline_left=self.points[0],
                far_baseline_right=self.points[1],
                near_baseline_left=self.points[2],
                near_baseline_right=self.points[3],
                far_service_left=self.points[4],
                far_service_right=self.points[5],
                near_service_left=self.points[6],
                near_service_right=self.points[7],
                net_left=self.points[8],
                net_right=self.points[9],
                far_center_t=self.points[10],
                near_center_t=self.points[11],
            )
            
            # Compute and save calibration
            self.calibrator.calibrate_from_keypoints(keypoints)
            self.calibrator.save_calibration(output_path)
            
            print(f"\n✓ Calibration saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"\nError saving calibration: {e}")
            return False
    
    def run(self, output_path: str) -> bool:
        """Run interactive calibration."""
        window_name = "Court Calibration - Click keypoints in order"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        print("\nInteractive Court Calibration")
        print("=" * 60)
        print("\nClick keypoints in this order:")
        for i, name in enumerate(self.POINT_NAMES, 1):
            print(f"  {i}. {name}")
        print("\nControls:")
        print("  Left click: Add point")
        print("  's': Save calibration")
        print("  'r': Reset points")
        print("  'q': Quit without saving")
        print("=" * 60)
        
        while True:
            cv2.imshow(window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nCalibration cancelled")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('r'):
                print("\nPoints reset")
                self.reset()
            
            elif key == ord('s'):
                success = self.save_calibration(output_path)
                cv2.destroyAllWindows()
                return success
        
        cv2.destroyAllWindows()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tennis court calibration tool"
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to tennis video"
    )
    
    parser.add_argument(
        "--frame",
        type=int,
        default=50,
        help="Frame number to use for calibration (default: 50)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/court_calibration.json",
        help="Output calibration file path"
    )
    
    args = parser.parse_args()
    
    # Load video
    print(f"\nLoading video: {args.video}")
    try:
        loader = VideoLoader(args.video)
        metadata = loader.get_metadata()
        print(f"Resolution: {metadata['width']}x{metadata['height']}")
        print(f"Total frames: {metadata['frame_count']}")
    except Exception as e:
        print(f"Error loading video: {e}")
        return 1
    
    # Extract calibration frame
    print(f"Extracting frame {args.frame}...")
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"Error: Could not read frame {args.frame}")
        return 1
    
    print(f"Frame extracted: {frame.shape}")
    
    # Run interactive calibration
    calibrator = InteractiveCalibrator(frame)
    success = calibrator.run(args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
