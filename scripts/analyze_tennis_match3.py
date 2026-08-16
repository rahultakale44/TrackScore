#!/usr/bin/env python3
"""
Analyze tennis_match3.mp4 to understand footage characteristics.

Determines:
- Resolution and FPS
- Camera type (static/moving)
- Court visibility
- Player positions
- Officials/ball boy positions
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_video():
    video_path = "samples/tennis_match3.mp4"
    
    print("=" * 70)
    print("tennis_match3.mp4 Video Analysis")
    print("=" * 70)
    
    cap = cv2.VideoCapture(video_path)
    
    # Metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\nVideo Metadata:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f}s")
    
    # Sample frames for analysis
    sample_frames = [0, 30, 90, 150, 210, 270, 330]
    
    print(f"\nAnalyzing {len(sample_frames)} sample frames...")
    
    # Detect camera movement
    prev_frame_gray = None
    motion_scores = []
    
    for idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame_gray is not None:
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame_gray, gray)
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
        
        prev_frame_gray = gray
    
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    
    print(f"\nCamera Analysis:")
    print(f"  Average frame difference: {avg_motion:.2f}")
    if avg_motion < 5:
        print(f"  Camera type: STATIC (very stable)")
    elif avg_motion < 15:
        print(f"  Camera type: MOSTLY STATIC (slight movement)")
    else:
        print(f"  Camera type: MOVING (tracking/panning)")
    
    # Extract first frame for detailed analysis
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, analysis_frame = cap.read()
    
    if ret:
        # Analyze court visibility
        # Convert to HSV and look for court colors (typically brown/green)
        hsv = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2HSV)
        
        # Look for white lines (court markings)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_percentage = (np.count_nonzero(white_mask) / white_mask.size) * 100
        
        print(f"\nCourt Visibility:")
        print(f"  White line pixels: {white_percentage:.2f}%")
        
        # Detect regions with people
        # Top 25% of frame = far court / spectators / officials
        # Middle 50% = playing court
        # Bottom 25% = near court / ball boys
        
        h, w = analysis_frame.shape[:2]
        
        regions = {
            "top_quarter": (0, h//4),
            "upper_middle": (h//4, h//2),
            "lower_middle": (h//2, 3*h//4),
            "bottom_quarter": (3*h//4, h),
        }
        
        print(f"\nFrame Regions (height {h}):")
        for name, (y1, y2) in regions.items():
            print(f"  {name}: y={y1}-{y2}")
        
        # Save annotated frame
        output_path = Path("outputs/inspection/tennis_match3_analysis.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Draw region boundaries
        annotated = analysis_frame.copy()
        for y in [h//4, h//2, 3*h//4]:
            cv2.line(annotated, (0, y), (w, y), (0, 255, 255), 2)
        
        # Add labels
        cv2.putText(annotated, "TOP QUARTER (far court/officials)", (10, h//8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, "UPPER MIDDLE (Player A zone)", (10, 3*h//8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, "LOWER MIDDLE (Player B zone)", (10, 5*h//8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, "BOTTOM QUARTER (near court/ball boys)", (10, 7*h//8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imwrite(str(output_path), annotated)
        print(f"\nAnnotated frame saved: {output_path}")
    
    cap.release()
    
    print("\n" + "=" * 70)
    print("Analysis Summary:")
    print("=" * 70)
    print(f"Video: tennis_match3.mp4")
    print(f"Resolution: {width}x{height} @ {fps:.0f}fps")
    print(f"Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"Camera: {'STATIC' if avg_motion < 15 else 'MOVING'}")
    print(f"\nNext Steps:")
    print(f"1. Manually calibrate court on this footage")
    print(f"2. Define court-aware player selection zones")
    print(f"3. Build robust player filtering")
    print("=" * 70)

if __name__ == "__main__":
    analyze_video()
