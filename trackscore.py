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
    
    # Preview window
    preview_window = "TrackScore - Processing"
    if not args.no_preview:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 1280, 720)
    
    # Storage for analytics
    all_frames_data = []
    calibration_attempted = False
    calibration_success = False
    
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
            
            # Detect and track ball  
            ball_tracking_result = ball_tracker.process_frame(frame, frame_number, timestamp)
            ball_tracked = ball_tracking_result.get("ball")
            
            # Attempt court calibration once early on
            if not calibration_attempted and frame_number > 10:
                try:
                    court_result = court_detector.detect_court_lines(frame)
                    lines = court_result.get("lines", [])
                    
                    if len(lines) >= 4:  # Need sufficient lines
                        # Attempt calibration
                        correspondences = []  # Would need proper court point mapping
                        # For now, skip calibration - would need manual or auto point detection
                        calibration_attempted = True
                        
                except Exception as e:
                    calibration_attempted = True
                    print(f"  Court calibration skipped: {e}")
            
            # Analyze ball trajectory
            if ball_tracked and isinstance(ball_tracked, dict):
                pos = ball_tracked.get("position", {})
                x = pos.get("x") if isinstance(pos, dict) else ball_tracked.get("x")
                y = pos.get("y") if isinstance(pos, dict) else ball_tracked.get("y")
                predicted = ball_tracked.get("predicted", False)
                
                if x is not None and y is not None:
                    ball_trajectory_analyzer.analyze_trajectory(x, y, predicted)
            
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
                bbox = player.get("bbox", [])
                label = player.get("label", "Player")
                if len(bbox) == 4:
                    player_info = {
                        "label": label,
                        "bbox": bbox,
                        "confidence": player.get("confidence", 0.0),
                    }
                    frame_data["players"].append(player_info)
            
            # Add ball data
            if ball_tracked and isinstance(ball_tracked, dict):
                pos = ball_tracked.get("position", {})
                x = pos.get("x") if isinstance(pos, dict) else ball_tracked.get("x")
                y = pos.get("y") if isinstance(pos, dict) else ball_tracked.get("y")
                
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
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n  Preview closed by user (q pressed)")
                        break
                
            except Exception as e:
                print(f"  Warning: Failed to render frame {frame_number}: {e}")
                out.write(frame)  # Write original frame
            
            # Store analytics
            all_frames_data.append(frame_data)
            
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
        if not args.no_preview:
            cv2.destroyAllWindows()
    
    elapsed_total = time.time() - start_time
    print(f"\n[4/6] Processing complete: {processed_count} frames in {elapsed_total:.1f}s")
    
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
