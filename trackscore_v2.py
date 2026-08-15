#!/usr/bin/env python3
"""
TrackScore v2 - Court-First Tennis Video Analytics

Redesigned perception pipeline:
1. Court calibration (manual one-time)
2. Court-aware player filtering
3. Court-aware ball detection
4. Enhanced debug visualization

Usage:
    python trackscore_v2.py samples/tennis_test2.mp4 --calibration data/court_calibration.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Core vision modules
from backend.app.vision.video_loader import VideoLoader
from backend.app.vision.player_detector import PlayerDetector
from backend.app.vision.player_tracker import PlayerTracker
from backend.app.vision.court_calibrator import CourtCalibrator
from backend.app.vision.court_aware_player_filter import CourtAwarePlayerFilter
from backend.app.vision.court_aware_ball_detector import CourtAwareBallDetector
from backend.app.vision.video_renderer import VideoRenderer, RendererConfig

# Scoring
from backend.app.scoring.tennis_scoring import TennisScoringEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TrackScore v2 - Court-First Tennis Video Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to input tennis video"
    )
    
    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Path to court calibration JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: outputs/final/trackscore_core_v2.mp4)"
    )
    
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Process only first N seconds"
    )
    
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable enhanced debug visualization"
    )
    
    return parser.parse_args()


def create_output_paths(video_path: str, output_arg: str = None):
    """Create output directory and paths."""
    if output_arg:
        output_video = Path(output_arg)
    else:
        output_video = Path("outputs/final/trackscore_core_v2.mp4")
    
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    output_dir = output_video.parent
    analytics_json = output_dir / "analytics_v2.json"
    summary_json = output_dir / "summary_v2.json"
    
    return {
        "video": output_video,
        "analytics": analytics_json,
        "summary": summary_json,
    }


def draw_debug_frame(
    frame: np.ndarray,
    court_calibrator: CourtCalibrator,
    filter_result: dict,
    ball_result: dict,
) -> np.ndarray:
    """
    Create enhanced debug visualization.
    
    Shows:
    - Court overlay (yellow)
    - All person detections (yellow boxes)
    - Rejected persons (red boxes with reason)
    - Selected Player A/B (green boxes)
    - Ball candidates (small orange circles)
    - Selected ball (bright green circle)
    - Predicted ball (blue circle)
    - Ball trajectory (colored line)
    """
    debug = frame.copy()
    
    # Draw court overlay
    debug = court_calibrator.draw_court_overlay(debug)
    
    # Draw all person detections in yellow
    for person in filter_result.get("all_persons", []):
        bbox = person.get("bbox", {})
        if isinstance(bbox, dict):
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 1)
    
    # Draw rejected persons in red with reason
    for person in filter_result.get("rejected_persons", []):
        bbox = person.get("bbox", {})
        reason = person.get("rejection_reason", "unknown")
        if isinstance(bbox, dict):
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                debug,
                f"REJECTED: {reason}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
    
    # Draw selected Player A/B in bright green
    for player in filter_result.get("player_candidates", []):
        bbox = player.get("bbox", {})
        label = player.get("player_label", "Player")
        court_side = player.get("court_side", "?")
        if isinstance(bbox, dict):
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                debug,
                f"{label} ({court_side})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
    
    # Draw ball ROI candidates in small orange
    for candidate in ball_result.get("roi_candidates", []):
        center = candidate.get("center", {})
        cx, cy = int(center.get("x", 0)), int(center.get("y", 0))
        cv2.circle(debug, (cx, cy), 3, (0, 165, 255), 1)
    
    # Draw selected ball
    selected_ball = ball_result.get("selected_ball")
    if selected_ball:
        is_predicted = selected_ball.get("predicted", False)
        filtered_pos = selected_ball.get("filtered_position", {})
        bx, by = int(filtered_pos.get("x", 0)), int(filtered_pos.get("y", 0))
        
        if is_predicted:
            # Blue for predicted
            cv2.circle(debug, (bx, by), 12, (255, 0, 0), 2)
            cv2.circle(debug, (bx, by), 4, (255, 0, 0), -1)
        else:
            # Bright green for detected
            cv2.circle(debug, (bx, by), 12, (0, 255, 0), 2)
            cv2.circle(debug, (bx, by), 4, (0, 255, 0), -1)
    
    # Draw ball trajectory
    trajectory = ball_result.get("trajectory", [])
    if len(trajectory) >= 2:
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]
            
            pt1 = (int(prev["x"]), int(prev["y"]))
            pt2 = (int(curr["x"]), int(curr["y"]))
            
            # Color based on predicted vs detected
            if curr["predicted"]:
                color = (255, 100, 0)  # Blue
            else:
                color = (0, 255, 255)  # Yellow
            
            cv2.line(debug, pt1, pt2, color, 2, cv2.LINE_AA)
    
    # Draw debug info text
    all_count = len(ball_result.get("all_candidates", []))
    roi_count = len(ball_result.get("roi_candidates", []))
    filtered_count = len(ball_result.get("filtered_candidates", []))
    ball_visible = ball_result.get("ball_visible", False)
    
    info_lines = [
        f"Ball candidates: {all_count} total, {roi_count} in ROI, {filtered_count} filtered",
        f"Ball visible: {ball_visible}",
        f"Persons: {len(filter_result.get('all_persons', []))} detected",
        f"Court persons: {len(filter_result.get('court_persons', []))}",
        f"Rejected: {len(filter_result.get('rejected_persons', []))}",
        f"Players: {len(filter_result.get('player_candidates', []))}",
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(
            debug,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y_offset += 20
    
    return debug


def process_video(args):
    """Main video processing pipeline."""
    print("=" * 70)
    print("TrackScore v2 - Court-First Tennis Video Analytics")
    print("=" * 70)
    print(f"\nInput video: {args.video}")
    print(f"Calibration: {args.calibration}")
    
    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        print(f"Error: Calibration file not found: {args.calibration}")
        return 1
    
    # Create output paths
    outputs = create_output_paths(args.video, args.output)
    print(f"Output video: {outputs['video']}")
    
    # Load video metadata
    print("\n[1/6] Loading video...")
    try:
        loader = VideoLoader(str(video_path))
        metadata = loader.get_metadata()
        
        fps = metadata["fps"]
        width = metadata["width"]
        height = metadata["height"]
        total_frames = metadata["frame_count"]
        duration = metadata["duration_seconds"]
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {duration:.2f}s ({total_frames} frames)")
        
        max_frames = None
        if args.max_seconds:
            max_frames = int(args.max_seconds * fps)
            print(f"  Processing limit: {args.max_seconds}s ({max_frames} frames)")
        
    except Exception as e:
        print(f"Error loading video: {e}")
        return 1
    
    # Load court calibration
    print("\n[2/6] Loading court calibration...")
    try:
        court_calibrator = CourtCalibrator()
        court_calibrator.load_calibration(str(calibration_path))
        print(f"  ✓ Court calibration loaded")
        print(f"  ✓ Homography matrix computed")
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return 1
    
    # Initialize detectors
    print("\n[3/6] Initializing detectors...")
    player_detector = PlayerDetector()
    player_tracker = PlayerTracker()
    player_filter = CourtAwarePlayerFilter(court_calibrator)
    ball_detector = CourtAwareBallDetector(court_calibrator)
    scoring_engine = TennisScoringEngine()
    
    print("  ✓ Player detector initialized")
    print("  ✓ Court-aware player filter initialized")
    print("  ✓ Court-aware ball detector initialized")
    
    # Open video for processing
    print("\n[4/6] Processing frames...")
    cap = cv2.VideoCapture(str(video_path))
    
    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(outputs["video"]),
        fourcc,
        fps,
        (width, height)
    )
    
    if not out.isOpened():
        print(f"Error: Failed to create output video writer")
        cap.release()
        return 1
    
    # Debug video writer
    debug_out = None
    if args.debug:
        debug_path = outputs["video"].parent / "trackscore_core_v2_debug.mp4"
        debug_out = cv2.VideoWriter(
            str(debug_path),
            fourcc,
            fps,
            (width, height)
        )
        if debug_out.isOpened():
            print(f"  Debug video: {debug_path}")
    
    # Preview windows
    preview_window = "TrackScore v2 - Processing"
    debug_window = "TrackScore v2 - Debug View"
    
    if not args.no_preview:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 1280, 720)
        
        if args.debug:
            cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(debug_window, 1280, 720)
    
    # Storage for analytics
    all_frames_data = []
    
    # Detection statistics
    stats = {
        "player_a_frames": 0,
        "player_b_frames": 0,
        "ball_detected_frames": 0,
        "ball_predicted_frames": 0,
        "total_frames": 0,
        "persons_rejected": 0,
    }
    
    frame_number = 0
    processed_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            if max_frames and processed_count >= max_frames:
                break
            
            timestamp = frame_number / fps
            
            # Detect persons with YOLO
            person_result = player_detector.detect_players(frame)
            persons_detected = person_result.get("players", [])
            
            # Track persons with ByteTrack
            tracking_result = player_tracker.process_frame(frame, frame_number, timestamp)
            tracked_persons = tracking_result.get("players", [])
            
            # Filter to actual tennis players using court coordinates
            filter_result = player_filter.filter_person_detections(tracked_persons)
            players = filter_result.get("player_candidates", [])
            
            # Detect ball with court awareness
            ball_result = ball_detector.detect_ball(frame, frame_number, timestamp)
            selected_ball = ball_result.get("selected_ball")
            ball_visible = ball_result.get("ball_visible", False)
            
            # Debug output
            if args.debug and frame_number % 30 == 0:
                labels = [p.get("player_label", "?") for p in players]
                rejection_reasons = {}
                for person in filter_result.get("rejected_persons", []):
                    reason = person.get("rejection_reason", "unknown")
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                
                print(f"  Frame {frame_number}: "
                      f"{len(persons_detected)} persons → "
                      f"{len(tracked_persons)} tracked → "
                      f"{len(filter_result.get('court_persons', []))} in court → "
                      f"{len(players)} players {labels}, "
                      f"ball_visible={ball_visible}")
                if rejection_reasons:
                    print(f"    Rejections: {rejection_reasons}")
            
            # Build frame analytics
            scoreboard = scoring_engine.get_live_scoreboard()
            
            frame_data = {
                "frame_number": frame_number,
                "timestamp_seconds": timestamp,
                "players": [],
                "ball": None,
                "events": [],
                "scoreboard": scoreboard,
            }
            
            # Add player data
            for player in players:
                bbox = player.get("bbox", {})
                label = player.get("player_label", "Player")
                
                if isinstance(bbox, dict):
                    bbox_list = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                else:
                    continue
                
                player_info = {
                    "label": label,
                    "bbox": bbox_list,
                    "confidence": player.get("confidence", 0.0),
                    "court_side": player.get("court_side", "unknown"),
                }
                frame_data["players"].append(player_info)
            
            # Add ball data
            if selected_ball:
                filtered_pos = selected_ball.get("filtered_position", {})
                frame_data["ball"] = {
                    "x": filtered_pos.get("x", 0),
                    "y": filtered_pos.get("y", 0),
                    "predicted": selected_ball.get("predicted", False),
                    "confidence": selected_ball.get("confidence", 0.0),
                }
            
            # Create debug frame
            if args.debug:
                debug_frame = draw_debug_frame(frame, court_calibrator, filter_result, ball_result)
                if debug_out and debug_out.isOpened():
                    debug_out.write(debug_frame)
            
            # Render final frame (simplified for now)
            output_frame = court_calibrator.draw_court_overlay(frame)
            
            # Draw players
            for player in players:
                bbox = player.get("bbox", {})
                label = player.get("player_label", "Player")
                if isinstance(bbox, dict):
                    x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                    x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw ball
            if selected_ball:
                filtered_pos = selected_ball.get("filtered_position", {})
                bx, by = int(filtered_pos.get("x", 0)), int(filtered_pos.get("y", 0))
                is_predicted = selected_ball.get("predicted", False)
                color = (255, 0, 0) if is_predicted else (0, 255, 0)
                cv2.circle(output_frame, (bx, by), 10, color, 2)
                cv2.circle(output_frame, (bx, by), 3, color, -1)
            
            out.write(output_frame)
            
            # Show preview
            if not args.no_preview:
                cv2.imshow(preview_window, output_frame)
                
                if args.debug and 'debug_frame' in locals():
                    cv2.imshow(debug_window, debug_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  Preview closed by user")
                    break
            
            # Store analytics
            all_frames_data.append(frame_data)
            
            # Update statistics
            stats["total_frames"] += 1
            stats["persons_rejected"] += len(filter_result.get("rejected_persons", []))
            
            for player in players:
                label = player.get("player_label", "")
                if "A" in label:
                    stats["player_a_frames"] += 1
                elif "B" in label:
                    stats["player_b_frames"] += 1
            
            if selected_ball:
                if selected_ball.get("predicted", False):
                    stats["ball_predicted_frames"] += 1
                else:
                    stats["ball_detected_frames"] += 1
            
            # Progress
            processed_count += 1
            if processed_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed_count} frames ({fps_rate:.1f} FPS)...")
            
            frame_number += 1
    
    finally:
        cap.release()
        out.release()
        if debug_out:
            debug_out.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    
    elapsed_total = time.time() - start_time
    print(f"\n[5/6] Processing complete: {processed_count} frames in {elapsed_total:.1f}s")
    
    # Print statistics
    print("\n  Detection Quality:")
    if stats["total_frames"] > 0:
        player_a_pct = (stats["player_a_frames"] / stats["total_frames"]) * 100
        player_b_pct = (stats["player_b_frames"] / stats["total_frames"]) * 100
        ball_det_pct = (stats["ball_detected_frames"] / stats["total_frames"]) * 100
        ball_pred_pct = (stats["ball_predicted_frames"] / stats["total_frames"]) * 100
        
        print(f"    Player A visible: {stats['player_a_frames']}/{stats['total_frames']} frames ({player_a_pct:.1f}%)")
        print(f"    Player B visible: {stats['player_b_frames']}/{stats['total_frames']} frames ({player_b_pct:.1f}%)")
        print(f"    Ball detected: {stats['ball_detected_frames']}/{stats['total_frames']} frames ({ball_det_pct:.1f}%)")
        print(f"    Ball predicted: {stats['ball_predicted_frames']}/{stats['total_frames']} frames ({ball_pred_pct:.1f}%)")
        print(f"    Persons rejected: {stats['persons_rejected']} total")
    
    # Save analytics
    print("\n[6/6] Saving analytics...")
    
    summary = {
        "input_video": str(video_path.absolute()),
        "output_video": str(outputs["video"].absolute()),
        "calibration_file": str(calibration_path.absolute()),
        "processed_frames": processed_count,
        "processing_time_seconds": round(elapsed_total, 2),
        "detection_stats": stats,
    }
    
    with open(outputs["summary"], 'w') as f:
        json.dump(summary, f, indent=2)
    
    analytics = {
        "video_metadata": metadata,
        "summary": summary,
        "frames": all_frames_data,
    }
    
    with open(outputs["analytics"], 'w') as f:
        json.dump(analytics, f, indent=2)
    
    print(f"  Summary saved: {outputs['summary']}")
    print(f"  Analytics saved: {outputs['analytics']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TrackScore v2 Analysis Complete")
    print("=" * 70)
    print(f"Input: {video_path}")
    print(f"Processed frames: {processed_count}/{total_frames}")
    print(f"Output video: {outputs['video']}")
    if args.debug:
        print(f"Debug video: {outputs['video'].parent / 'trackscore_core_v2_debug.mp4'}")
    print("=" * 70)
    
    return 0


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        return process_video(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
