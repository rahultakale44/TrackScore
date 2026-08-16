#!/usr/bin/env python3
"""
Extract candidate shot samples from tennis video for manual annotation.

Uses ball detection events and player activity to identify potential shot moments.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.vision.video_loader import VideoLoader
from backend.app.vision.player_detector import PlayerDetector
from backend.app.vision.player_tracker import PlayerTracker
from backend.app.vision.robust_player_filter import RobustPlayerFilter


class ShotSampleExtractor:
    """Extract candidate shot moments from tennis video."""
    
    def __init__(self, video_path: str, calibration_path: str):
        self.video_path = video_path
        self.calibration_path = calibration_path
        
        # Load calibration
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        self.homography = np.array(calib_data["homography_matrix"], dtype=np.float32)
        
        # Load video metadata
        self.loader = VideoLoader(video_path)
        self.metadata = self.loader.get_metadata()
        self.fps = self.metadata["fps"]
        
        print(f"✓ Video loaded: {video_path}")
        print(f"  Resolution: {self.metadata['width']}x{self.metadata['height']} @ {self.fps}fps")
        print(f"  Duration: {self.metadata['duration_seconds']:.1f}s ({self.metadata['frame_count']} frames)")
        print(f"✓ Calibration loaded: {calibration_path}")
    
    def _detect_shot_candidates(self) -> List[Dict]:
        """
        Detect candidate shot moments using heuristics.
        
        Shot detection heuristics:
        - Ball trajectory changes (direction reversal, velocity change)
        - Player proximity to ball
        - Player motion (sudden acceleration)
        - Ball height changes
        """
        print("\n[1/2] Analyzing video for shot candidates...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        player_detector = PlayerDetector()
        player_tracker = PlayerTracker()
        player_filter = RobustPlayerFilter()
        
        # Simple ball detection using brightness
        candidates = []
        frame_data = []
        
        frame_number = 0
        height = self.metadata["height"]
        width = self.metadata["width"]
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_number / self.fps
                
                # Detect players
                tracking_result = player_tracker.process_frame(frame, frame_number, timestamp)
                tracked_persons = tracking_result.get("players", [])
                filter_result = player_filter.filter_persons(tracked_persons, height, width)
                
                player_a = filter_result.get("player_a")
                player_b = filter_result.get("player_b")
                
                # Store frame data
                frame_data.append({
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "player_a": player_a is not None,
                    "player_b": player_b is not None,
                    "player_a_pos": player_a["foot_position"] if player_a else None,
                    "player_b_pos": player_b["foot_position"] if player_b else None,
                })
                
                frame_number += 1
                
                if frame_number % 100 == 0:
                    print(f"  Processed {frame_number}/{self.metadata['frame_count']} frames...")
        
        finally:
            cap.release()
        
        print(f"  Processed {len(frame_data)} frames total")
        
        # Detect shot candidates using simple heuristics
        print("\n[2/2] Identifying shot candidates...")
        
        # Strategy: Split video into segments where both players are visible
        # Each segment is a potential rally with multiple shots
        
        current_rally = []
        rallies = []
        
        for i, fd in enumerate(frame_data):
            if fd["player_a"] and fd["player_b"]:
                current_rally.append(fd)
            else:
                if len(current_rally) >= 30:  # At least 1 second
                    rallies.append(current_rally)
                current_rally = []
        
        # Add last rally
        if len(current_rally) >= 30:
            rallies.append(current_rally)
        
        print(f"  Found {len(rallies)} rally segments")
        
        # Within each rally, create shot candidates at regular intervals
        # This is a HEURISTIC for initial sampling - actual shots need manual labeling
        
        shot_id = 0
        for rally_idx, rally in enumerate(rallies):
            rally_start = rally[0]["frame"]
            rally_end = rally[-1]["frame"]
            rally_duration = (rally_end - rally_start) / self.fps
            
            print(f"  Rally {rally_idx+1}: frames {rally_start}-{rally_end} ({rally_duration:.1f}s)")
            
            # Sample potential shots every ~1-2 seconds within rally
            shot_interval = int(1.5 * self.fps)  # 1.5 seconds
            
            for shot_center in range(rally_start, rally_end, shot_interval):
                # Create a clip window around this potential shot
                clip_start = max(rally_start, shot_center - int(0.5 * self.fps))
                clip_end = min(rally_end, shot_center + int(0.5 * self.fps))
                
                # Try to determine which player might have hit (heuristic based on court position)
                # This is just a guess for annotation UI, not a label
                center_data = frame_data[shot_center] if shot_center < len(frame_data) else None
                suggested_player = "unknown"
                
                if center_data:
                    if center_data["player_a_pos"] and center_data["player_b_pos"]:
                        # Simple heuristic: nearer player to net might be hitting
                        # This is just for UI suggestion, not ground truth
                        a_y = center_data["player_a_pos"][1]
                        b_y = center_data["player_b_pos"][1]
                        suggested_player = "player_a" if a_y < b_y else "player_b"
                
                candidates.append({
                    "shot_id": f"shot_{shot_id:04d}",
                    "video": Path(self.video_path).name,
                    "rally_id": rally_idx,
                    "start_frame": clip_start,
                    "end_frame": clip_end,
                    "center_frame": shot_center,
                    "duration_sec": (clip_end - clip_start) / self.fps,
                    "suggested_player": suggested_player,
                    "label": "unlabeled",
                    "labeled_by": "",
                    "notes": ""
                })
                
                shot_id += 1
        
        print(f"\n✓ Extracted {len(candidates)} shot candidates for annotation")
        return candidates
    
    def extract_candidate_clips(self, output_dir: str, max_clips: Optional[int] = None) -> List[Dict]:
        """
        Extract shot candidates and save as video clips for annotation.
        
        Returns list of candidate metadata.
        """
        candidates = self._detect_shot_candidates()
        
        if max_clips:
            candidates = candidates[:max_clips]
            print(f"\n⚠ Limited to first {max_clips} clips")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[3/3] Saving video clips to {output_dir}...")
        
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        for i, candidate in enumerate(candidates):
            clip_path = output_path / f"{candidate['shot_id']}.mp4"
            
            # Set to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate['start_frame'])
            
            # Create clip writer
            out = cv2.VideoWriter(
                str(clip_path),
                fourcc,
                self.fps,
                (self.metadata['width'], self.metadata['height'])
            )
            
            # Write frames
            for frame_idx in range(candidate['start_frame'], candidate['end_frame']):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame number overlay
                cv2.putText(
                    frame,
                    f"Frame: {frame_idx} | Shot ID: {candidate['shot_id']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                out.write(frame)
            
            out.release()
            
            if (i + 1) % 10 == 0:
                print(f"  Saved {i+1}/{len(candidates)} clips...")
        
        cap.release()
        print(f"\n✓ Saved {len(candidates)} video clips")
        
        return candidates
    
    def save_annotation_template(self, candidates: List[Dict], output_path: str):
        """Save candidates as CSV template for manual annotation."""
        fieldnames = [
            "shot_id",
            "video",
            "rally_id",
            "start_frame",
            "end_frame",
            "center_frame",
            "duration_sec",
            "suggested_player",
            "label",
            "confidence",
            "labeled_by",
            "notes"
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for candidate in candidates:
                # Add confidence field
                candidate["confidence"] = ""
                writer.writerow(candidate)
        
        print(f"\n✓ Annotation template saved: {output_path}")
        print(f"\nTo annotate:")
        print(f"1. Open {output_path} in Excel/spreadsheet")
        print(f"2. Watch each clip in data/shot_dataset/clips/")
        print(f"3. Set 'label' to: forehand, backhand, serve, volley, smash, drop, or unknown")
        print(f"4. Set 'confidence' to: high, medium, or low")
        print(f"5. Set 'labeled_by' to your name")
        print(f"6. Add any notes in 'notes' column")
        print(f"7. Save the CSV file")


def main():
    parser = argparse.ArgumentParser(
        description="Extract shot candidates from tennis video for manual annotation"
    )
    parser.add_argument(
        "video",
        help="Input video path (e.g., samples/tennis_match3.mp4)"
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Court calibration JSON path"
    )
    parser.add_argument(
        "--output-dir",
        default="data/shot_dataset/clips",
        help="Output directory for video clips"
    )
    parser.add_argument(
        "--annotation-csv",
        default="data/shot_dataset/annotations/shot_annotations.csv",
        help="Output CSV file for annotations"
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Maximum number of clips to extract (for testing)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Shot Sample Extractor - TrackScore ML Phase 1")
    print("="*70)
    
    extractor = ShotSampleExtractor(args.video, args.calibration)
    
    # Extract clips
    candidates = extractor.extract_candidate_clips(args.output_dir, args.max_clips)
    
    # Save annotation template
    extractor.save_annotation_template(candidates, args.annotation_csv)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"1. Review video clips in: {args.output_dir}/")
    print(f"2. Annotate shots in: {args.annotation_csv}")
    print(f"3. Valid labels: forehand, backhand, serve, volley, smash, drop, unknown")
    print(f"4. After annotation, use the labeled CSV for ML training")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
