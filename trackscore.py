#!/usr/bin/env python3
"""
TrackScore - CLI Tennis Video Analytics

Usage:
    python trackscore.py path/to/video.mp4 [options]

Example:
    python trackscore.py samples/tennis_match.mp4
    python trackscore.py samples/tennis_match.mp4 --output outputs/my_analysis.mp4
    python trackscore.py samples/tennis_match.mp4 --max-seconds 10 --no-preview
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
from backend.app.vision.ball_detector import BallDetector
from backend.app.vision.ball_tracker import BallTracker
from backend.app.vision.court_line_detector import CourtLineDetector
from backend.app.vision.court_homography import CourtHomography
from backend.app.vision.video_renderer import VideoRenderer, RendererConfig

# Analytics modules
from backend.app.analytics.player_motion_analyzer import PlayerMotionAnalyzer
from backend.app.analytics.ball_trajectory_analyzer import BallTrajectoryAnalyzer
from backend.app.analytics.ball_speed_analyzer import BallSpeedAnalyzer
from backend.app.analytics.bounce_court_analyzer import BounceCourtAnalyzer

# ML and scoring
from backend.app.scoring.tennis_scoring import TennisScoringEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TrackScore - Tennis Video Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trackscore.py samples/tennis_match.mp4
  python trackscore.py video.mp4 --output outputs/analysis.mp4
  python trackscore.py video.mp4 --max-seconds 10 --no-preview
        """
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to input tennis video (MP4, MOV, AVI)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: outputs/final/trackscore_analysis.mp4)"
    )
    
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Process only first N seconds (for testing)"
    )
    
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview window during processing"
    )
    
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, process all frames)"
    )
    
    parser.add_argument(
        "--court-type",
        type=str,
        choices=["singles", "doubles"],
        default="singles",
        help="Court type for bounce detection (default: singles)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug visualizations showing detection internals"
    )
    
    return parser.parse_args()


def create_output_paths(video_path: str, output_arg: str = None):
    """Create output directory and determine output paths."""
    if output_arg:
        output_video = Path(output_arg)
    else:
        output_video = Path("outputs/final/trackscore_analysis.mp4")
    
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # Analytics and summary JSON files
    output_dir = output_video.parent
    video_stem = Path(video_path).stem
    analytics_json = output_dir / "analytics.json"
    summary_json = output_dir / "summary.json"
    
    return {
        "video": output_video,
        "analytics": analytics_json,
        "summary": summary_json,
    }


def process_video(args):
    """Main video processing pipeline."""
    print("=" * 70)
    print("TrackScore - Tennis Video Analytics")
    print("=" * 70)
    print(f"\nInput video: {args.video}")
    
    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1


def filter_court_lines(lines, frame_height, frame_width):
    """
    Filter raw Hough lines to main tennis court lines.
    
    Returns dominant horizontal and vertical lines that likely represent
    the tennis court structure.
    """
    if not lines:
        return []
    
    # Filter by length - court lines should be reasonably long
    # Reduce threshold from 0.15 to 0.10 to be less aggressive
    min_length = max(frame_width, frame_height) * 0.10
    long_lines = [l for l in lines if l.get("length_pixels", 0) > min_length]
    
    # Separate by orientation
    horizontal = [l for l in long_lines if l.get("orientation") == "horizontal"]
    vertical = [l for l in long_lines if l.get("orientation") == "vertical"]
    
    # Take strongest lines only (max ~6 horizontal, ~4 vertical for tennis court)
    horizontal_sorted = sorted(horizontal, key=lambda x: x.get("length_pixels", 0), reverse=True)
    vertical_sorted = sorted(vertical, key=lambda x: x.get("length_pixels", 0), reverse=True)
    
    main_lines = horizontal_sorted[:6] + vertical_sorted[:4]
    
    # Convert to simple pt1/pt2 format for rendering
    formatted_lines = []
    for line in main_lines:
        formatted_lines.append({
            "pt1": {"x": line["x1"], "y": line["y1"]},
            "pt2": {"x": line["x2"], "y": line["y2"]},
            "orientation": line.get("orientation", "unknown"),
            "length": line.get("length_pixels", 0),
        })
    
    return formatted_lines


def process_video(args):
    """Main video processing pipeline."""
    print("=" * 70)
    print("TrackScore - Tennis Video Analytics")
    print("=" * 70)
    print(f"\nInput video: {args.video}")
    
    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Create output paths
    outputs = create_output_paths(args.video, args.output)
    print(f"Output video: {outputs['video']}")
    print(f"Analytics JSON: {outputs['analytics']}")
    print(f"Summary JSON: {outputs['summary']}")
    
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
        
        # Calculate frame limit
        max_frames = None
        if args.max_seconds:
            max_frames = int(args.max_seconds * fps)
            print(f"  Processing limit: {args.max_seconds}s ({max_frames} frames)")
        
    except Exception as e:
        print(f"Error loading video: {e}")
        return 1
    
    # Initialize components
    print("\n[2/6] Initializing detectors and analyzers...")
    player_detector = PlayerDetector()
    player_tracker = PlayerTracker()
    ball_detector = BallDetector()
    ball_tracker = BallTracker()
    court_detector = CourtLineDetector()
    court_homography = CourtHomography()
    
    # Initialize analyzers (will be used if calibration succeeds)
    player_motion_analyzer = None
    ball_speed_analyzer = None
    bounce_analyzer = None
    ball_trajectory_analyzer = BallTrajectoryAnalyzer()
    
    # Initialize scoring
    scoring_engine = TennisScoringEngine()
    
    # Initialize renderer
    renderer_config = RendererConfig(
        output_path=str(outputs["video"]),
        show_scoreboard=True,
        show_ball_markers=True,
        show_trajectory=True,
        show_player_info=True,
        show_events=True,
    )
    renderer = VideoRenderer(renderer_config)
    
    # Open video for processing
    print("\n[3/6] Processing frames...")
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
        debug_path = outputs["video"].parent / "trackscore_core_debug.mp4"
        debug_out = cv2.VideoWriter(
            str(debug_path),
            fourcc,
            fps,
            (width, height)
        )
        if debug_out.isOpened():
            print(f"Debug video: {debug_path}")
        else:
            print(f"Warning: Failed to create debug video writer")
    
    # Preview window
    preview_window = "TrackScore - Processing"
    if not args.no_preview:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 1280, 720)
    
    # Debug window
    debug_window = "TrackScore - Debug View"
    if args.debug and not args.no_preview:
        cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(debug_window, 1280, 720)
    
    # Storage for analytics
    all_frames_data = []
    calibration_attempted = False
    calibration_success = False
    court_lines = []  # Store detected court lines
    
    # Detection statistics
    stats = {
        "player_a_frames": 0,
        "player_b_frames": 0,
        "ball_detected_frames": 0,
        "ball_predicted_frames": 0,
        "total_frames": 0,
    }
    
    frame_number = 0
    processed_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Apply frame stride
            if frame_number % args.frame_stride != 0:
                frame_number += 1
                continue
            
            # Check max frames limit
            if max_frames and processed_count >= max_frames:
                break
            
            timestamp = frame_number / fps
            
            # Detect and track players
            player_result = player_detector.detect_players(frame)
            players_detected = player_result.get("players", [])
            
            # Track players with correct API
            tracking_result = player_tracker.process_frame(frame, frame_number, timestamp)
            tracked_players = tracking_result.get("players", [])
            
            if args.debug and frame_number % 30 == 0:
                raw_count = tracking_result.get("raw_track_count", 0)
                court_count = tracking_result.get("court_track_count", 0)
                labels = [p.get("player_label", "?") for p in tracked_players]
                print(f"  Frame {frame_number}: {len(players_detected)} detected → {raw_count} raw → {court_count} in court → {len(tracked_players)} tracked {labels}")
            
            # Detect and track ball  
            ball_tracking_result = ball_tracker.process_frame(frame, frame_number, timestamp)
            ball_tracked = ball_tracking_result.get("ball")
            ball_visible = ball_tracking_result.get("ball_visible", False)
            
            if args.debug and frame_number % 30 == 0:
                print(f"  Frame {frame_number}: Ball visible={ball_visible}, tracked={ball_tracked is not None}")
            
            # Attempt court calibration once early on
            if not calibration_attempted and frame_number > 10 and frame_number < 100:
                try:
                    court_result = court_detector.analyse_frame(frame)
                    lines = court_result.get("lines", [])
                    
                    if args.debug and frame_number % 30 == 0:
                        print(f"  Frame {frame_number}: Detected {len(lines)} raw court lines")
                    
                    # Filter to main court lines
                    main_lines = filter_court_lines(lines, height, width)
                    
                    if len(main_lines) >= 6:  # Need at least 6 main lines for court
                        court_lines = main_lines
                        calibration_success = True
                        calibration_attempted = True
                        if args.debug:
                            print(f"  Court detected: {len(main_lines)} main lines from {len(lines)} raw candidates")
                    elif frame_number > 50:
                        calibration_attempted = True
                        if args.debug:
                            print(f"  Court detection incomplete: only {len(main_lines)} main lines found from {len(lines)} raw")
                    
                except Exception as e:
                    calibration_attempted = True
                    if args.debug:
                        print(f"  Court calibration failed: {e}")
            
            # Analyze ball trajectory
            if ball_tracked and isinstance(ball_tracked, dict):
                pos = ball_tracked.get("tracked_center", {})
                if not pos:
                    pos = ball_tracked.get("center", {})
                
                x = pos.get("x") if isinstance(pos, dict) else None
                y = pos.get("y") if isinstance(pos, dict) else None
                predicted = ball_tracked.get("predicted", False)
            
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
            for player in tracked_players:
                bbox = player.get("bbox", {})
                # PlayerTracker uses "player_label" not "label"
                label = player.get("player_label", player.get("label", "Player"))
                
                # Handle both dict and list bbox formats
                if isinstance(bbox, dict) and "x1" in bbox:
                    bbox_list = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                elif isinstance(bbox, list) and len(bbox) == 4:
                    bbox_list = bbox
                else:
                    continue  # Skip invalid bbox
                
                player_info = {
                    "label": label,
                    "bbox": bbox_list,
                    "confidence": player.get("confidence", 0.0),
                }
                frame_data["players"].append(player_info)
            
            # Add ball data
            if ball_tracked and isinstance(ball_tracked, dict):
                # Ball tracker returns "tracked_center" with x,y coordinates
                pos = ball_tracked.get("tracked_center", {})
                if not pos:
                    pos = ball_tracked.get("center", {})
                
                x = pos.get("x") if isinstance(pos, dict) else None
                y = pos.get("y") if isinstance(pos, dict) else None
                
                if x is not None and y is not None:
                    frame_data["ball"] = {
                        "x": x,
                        "y": y,
                        "predicted": ball_tracked.get("predicted", False),
                        "confidence": ball_tracked.get("confidence", 0.0),
                    }
            
            # Render frame
            try:
                annotated_frame = renderer.render_frame(frame, frame_data)
                out.write(annotated_frame)
                
                # Create debug frame if requested
                debug_frame = None
                if args.debug:
                    debug_frame = frame.copy()
                    
                    # Draw ALL detected players in debug
                    for i, player in enumerate(players_detected):
                        bbox = player.get("bbox", {})
                        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
                        conf = player.get("confidence", 0)
                        
                        # Draw in yellow for all detections
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(debug_frame, f"P{i} {conf:.2f}", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Draw tracked players in green
                    for player in tracked_players:
                        bbox = player.get("bbox", {})
                        if isinstance(bbox, dict):
                            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
                        else:
                            continue
                        label = player.get("player_label", player.get("label", "?"))
                        
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(debug_frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Draw ball candidates
                    ball_candidates = ball_tracking_result.get("yolo_candidate_count", 0)
                    cv2.putText(debug_frame, f"Ball candidates: {ball_candidates}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw tracked ball
                    if ball_tracked:
                        pos = ball_tracked.get("tracked_center", {})
                        x, y = int(pos.get("x", 0)), int(pos.get("y", 0))
                        cv2.circle(debug_frame, (x, y), 15, (0, 255, 0), 3)
                        cv2.circle(debug_frame, (x, y), 5, (0, 255, 0), -1)
                    
                    # Draw court lines if detected
                    if court_lines:
                        for line in court_lines:
                            pt1 = line.get("pt1", {})
                            pt2 = line.get("pt2", {})
                            if pt1 and pt2:
                                x1, y1 = int(pt1.get("x", 0)), int(pt1.get("y", 0))
                                x2, y2 = int(pt2.get("x", 0)), int(pt2.get("y", 0))
                                cv2.line(debug_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Write to debug video
                    if debug_out and debug_out.isOpened():
                        debug_out.write(debug_frame)
                
                # Show preview
                if not args.no_preview:
                    # Resize for preview if needed
                    preview_frame = annotated_frame
                    if width > 1920:
                        scale = 1920 / width
                        preview_frame = cv2.resize(
                            annotated_frame,
                            (int(width * scale), int(height * scale))
                        )
                    
                    cv2.imshow(preview_window, preview_frame)
                    
                    # Show debug window
                    if debug_frame is not None:
                        debug_preview = debug_frame
                        if width > 1920:
                            scale = 1920 / width
                            debug_preview = cv2.resize(
                                debug_frame,
                                (int(width * scale), int(height * scale))
                            )
                        cv2.imshow(debug_window, debug_preview)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n  Preview closed by user (q pressed)")
                        break
                
            except Exception as e:
                print(f"  Warning: Failed to render frame {frame_number}: {e}")
                out.write(frame)  # Write original frame
            
            # Store analytics
            all_frames_data.append(frame_data)
            
            # Update statistics
            stats["total_frames"] += 1
            for player in tracked_players:
                # PlayerTracker uses "player_label" field
                label = player.get("player_label", player.get("label", ""))
                if "A" in label:
                    stats["player_a_frames"] += 1
                elif "B" in label:
                    stats["player_b_frames"] += 1
            
            if ball_tracked:
                if ball_tracked.get("predicted", False):
                    stats["ball_predicted_frames"] += 1
                else:
                    stats["ball_detected_frames"] += 1
            
            # Progress update
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
    print(f"\n[4/6] Processing complete: {processed_count} frames in {elapsed_total:.1f}s")
    
    # Print detection statistics
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
    
    # Build analytics summary
    print("\n[5/6] Generating analytics...")
    
    # Count detections
    total_player_detections = sum(len(f["players"]) for f in all_frames_data)
    total_ball_detections = sum(1 for f in all_frames_data if f["ball"])
    
    # Get final scores
    final_scoreboard = scoring_engine.get_live_scoreboard()
    final_stats = scoring_engine.get_match_statistics()
    winner = final_scoreboard.get("winner")
    
    # Create summary
    summary = {
        "input_video": str(video_path.absolute()),
        "output_video": str(outputs["video"].absolute()),
        "processed_frames": processed_count,
        "processing_time_seconds": round(elapsed_total, 2),
        "processing_fps": round(processed_count / elapsed_total, 2) if elapsed_total > 0 else 0,
        "video_info": {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration_seconds": duration,
            "total_frames": total_frames,
        },
        "detections": {
            "player_detections": total_player_detections,
            "ball_detections": total_ball_detections,
        },
        "court_calibration": {
            "attempted": calibration_attempted,
            "success": calibration_success,
        },
        "match_score": final_scoreboard,
        "match_statistics": final_stats,
    }
    
    # Save summary JSON
    with open(outputs["summary"], 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save full analytics JSON
    analytics = {
        "video_metadata": metadata,
        "summary": summary,
        "frames": all_frames_data,
    }
    
    with open(outputs["analytics"], 'w') as f:
        json.dump(analytics, f, indent=2)
    
    print(f"  Summary saved: {outputs['summary']}")
    print(f"  Analytics saved: {outputs['analytics']}")
    
    # Final output verification
    print("\n[6/6] Verifying output...")
    if outputs["video"].exists():
        size_mb = outputs["video"].stat().st_size / (1024 * 1024)
        print(f"  Output video created: {size_mb:.2f} MB")
        
        # Verify video is playable
        verify_cap = cv2.VideoCapture(str(outputs["video"]))
        if verify_cap.isOpened():
            verify_fps = verify_cap.get(cv2.CAP_PROP_FPS)
            verify_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"  Video verification: {verify_frames} frames @ {verify_fps:.2f} FPS ✓")
            verify_cap.release()
        else:
            print(f"  Warning: Output video may not be playable")
    else:
        print(f"  Error: Output video was not created")
        return 1
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TrackScore Analysis Complete")
    print("=" * 70)
    print(f"Input: {video_path}")
    print(f"Processed frames: {processed_count}/{total_frames}")
    print(f"Processing time: {elapsed_total:.1f}s")
    print(f"Output video: {outputs['video']}")
    print(f"Analytics JSON: {outputs['analytics']}")
    print(f"Summary JSON: {outputs['summary']}")
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
