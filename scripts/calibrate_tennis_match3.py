#!/usr/bin/env python3
"""
UNAMBIGUOUS Tennis Court Calibration for tennis_match3.mp4

EXACT 14-POINT ORDER (no duplicates, no ambiguity):

Singles tennis court standard dimensions:
- Length: 23.77 m (baseline to baseline)
- Width: 8.23 m (singles sidelines)
- Service line: 6.40 m from net

Court coordinate system:
- Origin: center of NEAR baseline (bottom of screen)
- X-axis: left (-4.115m) to right (+4.115m)
- Y-axis: near (0m) to far (23.77m)

14 REQUIRED KEYPOINTS:
1.  FBL  = Far Baseline LEFT corner           → (-4.115, 23.77)
2.  FBR  = Far Baseline RIGHT corner          → (+4.115, 23.77)
3.  FSSL = Far Service line LEFT end          → (-4.115, 17.285)
4.  FSSR = Far Service line RIGHT end         → (+4.115, 17.285)
5.  FCSL = Far Center Service line T-junction → (0.0, 17.285)
6.  NL   = Net LEFT post                      → (-4.115, 11.885)
7.  NR   = Net RIGHT post                     → (+4.115, 11.885)
8.  NC   = Net CENTER                         → (0.0, 11.885)
9.  NCSL = Near Center Service line T-junction → (0.0, 6.485)
10. NSSL = Near Service line LEFT end         → (-4.115, 6.485)
11. NSSR = Near Service line RIGHT end        → (+4.115, 6.485)
12. NBL  = Near Baseline LEFT corner          → (-4.115, 0.0)
13. NBR  = Near Baseline RIGHT corner         → (+4.115, 0.0)
14. NBC  = Near Baseline CENTER               → (0.0, 0.0)

Controls:
  LEFT CLICK  = Mark current keypoint
  U           = Undo last point
  R           = Restart calibration
  S           = Save (only after validation)
  ESC         = Cancel
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class TennisCourtCalibrator:
    """Unambiguous tennis court calibration tool."""
    
    # Standard tennis court dimensions (singles)
    COURT_LENGTH = 23.77  # meters
    COURT_WIDTH = 8.23  # meters (singles)
    SERVICE_LINE_DIST = 6.40  # from net to service line
    
    # 14 keypoints with abbreviations
    KEYPOINTS = [
        ("FBL",  "Far Baseline LEFT corner"),
        ("FBR",  "Far Baseline RIGHT corner"),
        ("FSSL", "Far Service line LEFT end"),
        ("FSSR", "Far Service line RIGHT end"),
        ("FCSL", "Far Center Service line T-junction"),
        ("NL",   "Net LEFT post"),
        ("NR",   "Net RIGHT post"),
        ("NC",   "Net CENTER"),
        ("NCSL", "Near Center Service line T-junction"),
        ("NSSL", "Near Service line LEFT end"),
        ("NSSR", "Near Service line RIGHT end"),
        ("NBL",  "Near Baseline LEFT corner"),
        ("NBR",  "Near Baseline RIGHT corner"),
        ("NBC",  "Near Baseline CENTER"),
    ]
    
    def __init__(self, frame):
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.display = frame.copy()
        self.points = []
        self.homography = None
        self.reprojection_error = None
        
        # Create side panel
        self.panel_width = 500
        self.full_display = None
        
    def get_canonical_court_points(self):
        """
        Get canonical court coordinates matching the 14 keypoints.
        
        Coordinate system:
        - Origin at center of near baseline
        - X: left (-) to right (+)
        - Y: near (0) to far (+)
        """
        hw = self.COURT_WIDTH / 2.0  # half width = 4.115
        net_y = self.COURT_LENGTH / 2.0  # 11.885
        service_near_y = net_y - self.SERVICE_LINE_DIST  # 5.485
        service_far_y = net_y + self.SERVICE_LINE_DIST  # 18.285
        
        # IMPORTANT: Order must match KEYPOINTS list exactly
        canonical = [
            (-hw, self.COURT_LENGTH),  # 0: FBL
            (hw, self.COURT_LENGTH),   # 1: FBR
            (-hw, service_far_y),      # 2: FSSL
            (hw, service_far_y),       # 3: FSSR
            (0.0, service_far_y),      # 4: FCSL
            (-hw, net_y),              # 5: NL
            (hw, net_y),               # 6: NR
            (0.0, net_y),              # 7: NC
            (0.0, service_near_y),     # 8: NCSL
            (-hw, service_near_y),     # 9: NSSL
            (hw, service_near_y),      # 10: NSSR
            (-hw, 0.0),                # 11: NBL
            (hw, 0.0),                 # 12: NBR
            (0.0, 0.0),                # 13: NBC
        ]
        
        return canonical
    
    def compute_homography(self):
        """Compute homography with RANSAC."""
        if len(self.points) < 4:
            return False
        
        image_pts = np.array(self.points, dtype=np.float32)
        canonical = self.get_canonical_court_points()
        court_pts = np.array(canonical[:len(self.points)], dtype=np.float32)
        
        H, mask = cv2.findHomography(image_pts, court_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if H is None:
            return False
        
        self.homography = H
        
        # Calculate reprojection error
        projected = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H)
        errors = np.sqrt(np.sum((projected.reshape(-1, 2) - court_pts)**2, axis=1))
        self.reprojection_error = np.mean(errors)
        
        return True
    
    def draw_projected_court(self, canvas):
        """Draw the complete canonical tennis court projected onto image."""
        if self.homography is None or len(self.points) < 4:
            return
        
        # Get all canonical points
        canonical = self.get_canonical_court_points()
        
        # Project back to image coordinates
        H_inv = np.linalg.inv(self.homography)
        
        def project_point(court_pt):
            """Project single court point to image coordinates."""
            pt = np.array([[court_pt[0], court_pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
            img_pt = cv2.perspectiveTransform(pt, H_inv)
            return (int(img_pt[0, 0, 0]), int(img_pt[0, 0, 1]))
        
        # Define court lines
        lines_to_draw = [
            # Far baseline
            (canonical[0], canonical[1], "Far Baseline"),
            # Near baseline
            (canonical[11], canonical[12], "Near Baseline"),
            # Left sideline
            (canonical[0], canonical[11], "Left Sideline"),
            # Right sideline
            (canonical[1], canonical[12], "Right Sideline"),
            # Far service line
            (canonical[2], canonical[3], "Far Service"),
            # Near service line
            (canonical[9], canonical[10], "Near Service"),
            # Net
            (canonical[5], canonical[6], "Net"),
            # Center service line
            (canonical[4], canonical[8], "Center Service"),
        ]
        
        # Draw lines
        for pt1, pt2, name in lines_to_draw:
            try:
                img_pt1 = project_point(pt1)
                img_pt2 = project_point(pt2)
                cv2.line(canvas, img_pt1, img_pt2, (0, 255, 255), 2, cv2.LINE_AA)
            except:
                pass  # Skip if projection fails
    
    def create_side_panel(self):
        """Create info panel showing keypoint list and status."""
        panel = np.zeros((self.h, self.panel_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "COURT CALIBRATION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(panel, f"tennis_match3.mp4", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Progress
        cv2.putText(panel, f"Point {len(self.points)}/14", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Keypoint list
        y_offset = 120
        for i, (abbr, desc) in enumerate(self.KEYPOINTS):
            if i < len(self.points):
                # Completed point - green
                color = (0, 255, 0)
                status = "✓"
            elif i == len(self.points):
                # Current point - yellow
                color = (0, 255, 255)
                status = "→"
            else:
                # Future point - gray
                color = (100, 100, 100)
                status = " "
            
            # Abbreviation
            cv2.putText(panel, f"{status} {abbr:4s}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 if i >= len(self.points) else 2)
            
            # Description (smaller)
            cv2.putText(panel, desc[:30], (80, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            # Show clicked coordinates if available
            if i < len(self.points):
                coord_text = f"({int(self.points[i][0])},{int(self.points[i][1])})"
                cv2.putText(panel, coord_text, (80, y_offset + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            
            y_offset += 35
        
        # Controls
        y_offset += 20
        cv2.rectangle(panel, (5, y_offset), (self.panel_width-5, y_offset+100), (50, 50, 50), -1)
        y_offset += 20
        cv2.putText(panel, "CONTROLS:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 25
        controls = [
            "Click = Mark point",
            "U = Undo last",
            "R = Restart",
            "S = Save",
            "ESC = Cancel"
        ]
        for ctrl in controls:
            cv2.putText(panel, ctrl, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        # Reprojection error if available
        if self.reprojection_error is not None:
            y_offset += 10
            error_color = (0, 255, 0) if self.reprojection_error < 0.5 else (0, 165, 255) if self.reprojection_error < 1.0 else (0, 0, 255)
            cv2.putText(panel, f"Error: {self.reprojection_error:.3f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 2)
        
        return panel
    
    def update_display(self):
        """Update complete display with frame + panel."""
        # Start with original frame
        frame_canvas = self.frame.copy()
        
        # Draw current instruction on frame
        if len(self.points) < len(self.KEYPOINTS):
            abbr, desc = self.KEYPOINTS[len(self.points)]
            instruction = f"Click: {abbr} - {desc}"
            
            # Black background box
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame_canvas, (5, 5), (text_size[0] + 15, 45), (0, 0, 0), -1)
            cv2.putText(frame_canvas, f"Point {len(self.points)+1}/14", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame_canvas, instruction, (10, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw clicked points with labels
        for i, pt in enumerate(self.points):
            abbr, _ = self.KEYPOINTS[i]
            cv2.circle(frame_canvas, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)
            cv2.putText(frame_canvas, abbr, (int(pt[0])+10, int(pt[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw projected court if enough points
        if len(self.points) >= 4:
            self.draw_projected_court(frame_canvas)
        
        # Create side panel
        panel = self.create_side_panel()
        
        # Combine frame and panel
        self.full_display = np.hstack([frame_canvas, panel])
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only accept clicks on frame area (not panel)
            if x < self.w and len(self.points) < len(self.KEYPOINTS):
                self.points.append((float(x), float(y)))
                
                # Print correspondence
                abbr, desc = self.KEYPOINTS[len(self.points)-1]
                canonical = self.get_canonical_court_points()[len(self.points)-1]
                print(f"\n✓ Point {len(self.points)}/14: {abbr:4s} - {desc}")
                print(f"  Image: ({x}, {y})")
                print(f"  Court: ({canonical[0]:.3f}, {canonical[1]:.3f}) meters")
                
                # Compute homography if enough points
                if len(self.points) >= 4:
                    if self.compute_homography():
                        print(f"  Reprojection error: {self.reprojection_error:.3f} meters")
                
                self.update_display()
    
    def undo_last(self):
        """Remove last clicked point."""
        if self.points:
            removed = self.points.pop()
            abbr, desc = self.KEYPOINTS[len(self.points)]
            print(f"\n↶ Undone: {abbr} at ({int(removed[0])}, {int(removed[1])})")
            
            # Recompute homography
            if len(self.points) >= 4:
                self.compute_homography()
            else:
                self.homography = None
                self.reprojection_error = None
            
            self.update_display()
    
    def reset(self):
        """Clear all points."""
        self.points = []
        self.homography = None
        self.reprojection_error = None
        print("\n⟲ Calibration reset - all points cleared")
        self.update_display()
    
    def validate_and_save(self, output_path):
        """Validate calibration and save if acceptable."""
        if len(self.points) != len(self.KEYPOINTS):
            print(f"\n✗ ERROR: Need all {len(self.KEYPOINTS)} points, only have {len(self.points)}")
            return False
        
        if self.homography is None:
            self.compute_homography()
        
        if self.homography is None:
            print("\n✗ ERROR: Could not compute homography")
            return False
        
        # Validation thresholds
        MAX_ACCEPTABLE_ERROR = 1.5  # meters
        
        if self.reprojection_error > MAX_ACCEPTABLE_ERROR:
            print(f"\n✗ ERROR: Reprojection error too high: {self.reprojection_error:.3f}m")
            print(f"  Maximum acceptable: {MAX_ACCEPTABLE_ERROR}m")
            print(f"  Please review keypoint placement and recalibrate")
            return False
        
        # Save calibration
        canonical = self.get_canonical_court_points()
        
        data = {
            "video": "tennis_match3.mp4",
            "resolution": f"{self.w}x{self.h}",
            "keypoints": [
                {
                    "index": i,
                    "abbr": abbr,
                    "name": desc,
                    "image_coords": list(self.points[i]),
                    "court_coords": list(canonical[i])
                }
                for i, (abbr, desc) in enumerate(self.KEYPOINTS)
            ],
            "homography_matrix": self.homography.tolist(),
            "reprojection_error_meters": float(self.reprojection_error),
            "court_dimensions": {
                "length_m": self.COURT_LENGTH,
                "width_m": self.COURT_WIDTH,
                "type": "singles"
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✓ CALIBRATION SAVED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Output: {output_path}")
        print(f"Points: {len(self.points)}")
        print(f"Reprojection error: {self.reprojection_error:.3f} meters ✓")
        print(f"{'='*70}")
        
        return True
    
    def run(self, output_path):
        """Run interactive calibration."""
        window = "Tennis Court Calibration"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self.w + self.panel_width, self.h)
        cv2.setMouseCallback(window, self.mouse_callback)
        
        # Print complete keypoint list
        print("\n" + "="*70)
        print("TENNIS COURT CALIBRATION - tennis_match3.mp4")
        print("="*70)
        print("\nREQUIRED 14 KEYPOINTS (in exact order):\n")
        for i, (abbr, desc) in enumerate(self.KEYPOINTS, 1):
            canonical = self.get_canonical_court_points()[i-1]
            print(f"  {i:2d}. {abbr:4s} = {desc:45s} → ({canonical[0]:+7.3f}, {canonical[1]:6.3f})")
        
        print("\nCONTROLS:")
        print("  LEFT CLICK = Mark current keypoint")
        print("  U          = Undo last point")
        print("  R          = Restart calibration")
        print("  S          = Save (validates first)")
        print("  ESC        = Cancel")
        print("="*70 + "\n")
        
        self.update_display()
        
        while True:
            cv2.imshow(window, self.full_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n✗ Calibration cancelled")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('u') or key == ord('U'):
                self.undo_last()
            
            elif key == ord('r') or key == ord('R'):
                self.reset()
            
            elif key == ord('s') or key == ord('S'):
                success = self.validate_and_save(output_path)
                if success:
                    cv2.destroyAllWindows()
                    return True
        
        cv2.destroyAllWindows()
        return False


def main():
    video_path = "samples/tennis_match3.mp4"
    output_path = "data/calibration/tennis_match3.json"
    
    # Extract calibration frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("✗ ERROR: Could not read frame from video")
        return 1
    
    print(f"\nUsing frame 50 from {video_path}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run calibration
    calibrator = TennisCourtCalibrator(frame)
    success = calibrator.run(output_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
