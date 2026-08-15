#!/usr/bin/env python3
"""
Audit what the actual detectors return on real video frames.
"""

import cv2
import json
from pathlib import Path

from backend.app.vision.video_loader import VideoLoader
from backend.app.vision.player_detector import PlayerDetector
from backend.app.vision.player_tracker import PlayerTracker
from backend.app.vision.ball_detector import BallDetector
from backend.app.vision.ball_tracker import BallTracker
from backend.app.vision.court_line_detector import CourtLineDetector

def audit_frame_30():
    """Audit frame 30 of the sample video."""
    video_path = "samples/tennis_match.mp4"
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Skip to frame 30 (should have some action)
    for _ in range(30):
        ret, frame = cap.read()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame 30")
        return
    
    print("Frame 30 loaded: {}x{}".format(frame.shape[1], frame.shape[0]))
    
    # Test player detector
    print("\n=== PLAYER DETECTOR ===")
    player_detector = PlayerDetector()
    player_result = player_detector.detect_players(frame)
    print(f"Result keys: {list(player_result.keys())}")
    players = player_result.get("players", [])
    print(f"Players detected: {len(players)}")
    for i, player in enumerate(players[:5]):  # Show first 5
        bbox = player.get("bbox", [])
        conf = player.get("confidence", 0)
        print(f"  Player {i}: bbox={bbox}, conf={conf:.3f}")
    
    # Test player tracker
    print("\n=== PLAYER TRACKER ===")
    player_tracker = PlayerTracker()
    tracking_result = player_tracker.process_frame(frame, 30, 1.0)
    print(f"Result keys: {list(tracking_result.keys())}")
    tracked = tracking_result.get("players", [])
    print(f"Tracked players: {len(tracked)}")
    for i, player in enumerate(tracked):
        label = player.get("label", "?")
        bbox = player.get("bbox", [])
        print(f"  {label}: bbox={bbox}")
    
    # Test ball detector
    print("\n=== BALL DETECTOR ===")
    ball_detector = BallDetector()
    ball_result = ball_detector.detect_ball(frame)
    print(f"Result keys: {list(ball_result.keys())}")
    candidates = ball_result.get("candidates", [])
    print(f"Ball candidates: {len(candidates)}")
    for i, cand in enumerate(candidates[:3]):
        x, y = cand.get("x", 0), cand.get("y", 0)
        conf = cand.get("confidence", 0)
        print(f"  Candidate {i}: ({x}, {y}), conf={conf:.3f}")
    
    # Test ball tracker
    print("\n=== BALL TRACKER ===")
    ball_tracker = BallTracker()
    ball_track_result = ball_tracker.process_frame(frame, 30, 1.0)
    print(f"Result keys: {list(ball_track_result.keys())}")
    ball = ball_track_result.get("ball")
    if ball:
        print(f"Ball tracked: {ball}")
    else:
        print("Ball: None")
    
    # Test court detector
    print("\n=== COURT LINE DETECTOR ===")
    court_detector = CourtLineDetector()
    court_result = court_detector.analyse_frame(frame)
    print(f"Result keys: {list(court_result.keys())}")
    lines = court_result.get("lines", [])
    print(f"Lines detected: {len(lines)}")
    horizontal = [l for l in lines if l.get("orientation") == "horizontal"]
    vertical = [l for l in lines if l.get("orientation") == "vertical"]
    print(f"  Horizontal: {len(horizontal)}")
    print(f"  Vertical: {len(vertical)}")
    
    cap.release()
    
    print("\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    audit_frame_30()
