"""
Professional video renderer with analytics overlays.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class VideoRendererError(Exception):
    """Raised when video rendering fails."""


@dataclass
class RendererConfig:
    """Configuration for video renderer."""
    
    output_path: str = "output_annotated.mp4"
    codec: str = "mp4v"  # Browser-compatible codec
    quality: int = 90
    
    # Overlay settings
    show_scoreboard: bool = True
    show_ball_markers: bool = True
    show_trajectory: bool = True
    show_player_info: bool = True
    show_events: bool = True
    
    # Trajectory settings
    trajectory_length: int = 30  # Number of previous positions to show
    
    # Font settings
    font_scale_base: float = 0.6
    font_thickness_base: int = 2
    
    # Colors (BGR format)
    primary_color: Tuple[int, int, int] = (237, 99, 37)  # Blue
    ball_detected_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    ball_predicted_color: Tuple[int, int, int] = (0, 165, 255)  # Orange
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    event_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow


class VideoRenderer:
    """
    Renders analytics overlays on tennis match videos.
    
    Features:
    - TrackScore branding
    - Live scoreboard
    - Ball markers with trajectory
    - Player information
    - Event overlays
    - Court geometry
    """
    
    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config if config is not None else RendererConfig()
        self.trajectory_buffer: List[Dict[str, Any]] = []
    
    def _calculate_font_scale(self, frame_height: int) -> float:
        """Calculate font scale based on frame resolution."""
        base_height = 720
        return self.config.font_scale_base * (frame_height / base_height)
    
    def _calculate_font_thickness(self, frame_height: int) -> int:
        """Calculate font thickness based on frame resolution."""
        base_height = 720
        scale = frame_height / base_height
        return max(1, int(self.config.font_thickness_base * scale))
    
    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float,
        font_thickness: int,
        text_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
        padding: int = 5,
    ) -> None:
        """Draw text with a background rectangle."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1,
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )
    
    def _draw_header(
        self,
        frame: np.ndarray,
        timestamp: float,
        font_scale: float,
        font_thickness: int,
    ) -> None:
        """Draw TrackScore branding and timestamp in top-left."""
        height, width = frame.shape[:2]
        
        # Draw TrackScore logo
        logo_text = "TrackScore"
        self._draw_text_with_background(
            frame,
            logo_text,
            (20, 40),
            font_scale * 1.2,
            font_thickness + 1,
            self.config.text_color,
            self.config.primary_color,
            padding=8,
        )
        
        # Draw timestamp
        time_text = f"Time: {timestamp:.2f}s"
        self._draw_text_with_background(
            frame,
            time_text,
            (20, 80),
            font_scale * 0.8,
            font_thickness,
            self.config.text_color,
            self.config.background_color,
        )
    
    def _draw_scoreboard(
        self,
        frame: np.ndarray,
        scoreboard: Optional[Dict[str, Any]],
        font_scale: float,
        font_thickness: int,
    ) -> None:
        """Draw live scoreboard in top-right."""
        if not scoreboard or not self.config.show_scoreboard:
            return
        
        height, width = frame.shape[:2]
        
        # Extract scores
        points = scoreboard.get("points", {})
        games = scoreboard.get("games", {"Player A": 0, "Player B": 0})
        sets = scoreboard.get("sets", {"Player A": 0, "Player B": 0})
        
        # Scoreboard position (top-right)
        board_x = width - 250
        board_y = 30
        
        # Draw background
        cv2.rectangle(
            frame,
            (board_x - 10, board_y - 10),
            (width - 10, board_y + 120),
            self.config.background_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (board_x - 10, board_y - 10),
            (width - 10, board_y + 120),
            self.config.primary_color,
            2,
        )
        
        # Draw header
        cv2.putText(
            frame,
            "SCORE",
            (board_x, board_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.7,
            self.config.text_color,
            font_thickness,
            cv2.LINE_AA,
        )
        
        # Draw Player A
        player_a_text = f"A: {points.get('Player A', '0')} | {games['Player A']} | {sets['Player A']}"
        cv2.putText(
            frame,
            player_a_text,
            (board_x, board_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.6,
            self.config.text_color,
            font_thickness - 1,
            cv2.LINE_AA,
        )
        
        # Draw Player B
        player_b_text = f"B: {points.get('Player B', '0')} | {games['Player B']} | {sets['Player B']}"
        cv2.putText(
            frame,
            player_b_text,
            (board_x, board_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.6,
            self.config.text_color,
            font_thickness - 1,
            cv2.LINE_AA,
        )
        
        # Draw winner if available
        winner = scoreboard.get("winner")
        if winner:
            cv2.putText(
                frame,
                f"Winner: {winner}",
                (board_x, board_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.6,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA,
            )
    
    def _draw_ball_marker(
        self,
        frame: np.ndarray,
        ball: Dict[str, Any],
        font_scale: float,
    ) -> None:
        """Draw ball marker with predicted/detected state."""
        if not self.config.show_ball_markers:
            return
        
        x = int(ball.get("x", 0))
        y = int(ball.get("y", 0))
        predicted = ball.get("predicted", False)
        
        # Choose color based on detection state
        color = self.config.ball_predicted_color if predicted else self.config.ball_detected_color
        
        # Draw circle marker
        cv2.circle(frame, (x, y), 8, color, 2)
        cv2.circle(frame, (x, y), 3, color, -1)
        
        # Add label for predicted balls
        if predicted:
            label = "Est."
            cv2.putText(
                frame,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    
    def _draw_trajectory(self, frame: np.ndarray) -> None:
        """Draw ball trajectory tail."""
        if not self.config.show_trajectory or len(self.trajectory_buffer) < 2:
            return
        
        # Draw lines connecting trajectory points
        for i in range(len(self.trajectory_buffer) - 1):
            pt1 = self.trajectory_buffer[i]
            pt2 = self.trajectory_buffer[i + 1]
            
            if "x" in pt1 and "x" in pt2:
                x1, y1 = int(pt1["x"]), int(pt1["y"])
                x2, y2 = int(pt2["x"]), int(pt2["y"])
                
                # Fade older points
                alpha = (i + 1) / len(self.trajectory_buffer)
                color = tuple(int(c * alpha) for c in self.config.ball_detected_color)
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    def _draw_player_info(
        self,
        frame: np.ndarray,
        players: List[Dict[str, Any]],
        font_scale: float,
        font_thickness: int,
    ) -> None:
        """Draw player information and speed estimates."""
        if not self.config.show_player_info:
            return
        
        for player in players:
            bbox = player.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.primary_color, 2)
            
            # Draw player label
            label = player.get("label", "Player")
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.5,
                self.config.primary_color,
                font_thickness,
                cv2.LINE_AA,
            )
            
            # Draw speed if available
            speed = player.get("speed_kmh")
            if speed is not None:
                speed_text = f"~{speed:.1f} km/h"
                cv2.putText(
                    frame,
                    speed_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.4,
                    self.config.text_color,
                    font_thickness - 1,
                    cv2.LINE_AA,
                )
    
    def _draw_event_overlay(
        self,
        frame: np.ndarray,
        event: Dict[str, Any],
        font_scale: float,
        font_thickness: int,
    ) -> None:
        """Draw event overlay (BOUNCE, IN, OUT, SHOT, RALLY)."""
        if not self.config.show_events:
            return
        
        height, width = frame.shape[:2]
        
        event_type = event.get("type", "").upper()
        description = event.get("description", "")
        
        if not event_type:
            return
        
        # Center position
        text = event_type
        if description:
            text = f"{event_type}: {description}"
        
        # Calculate text size for centering
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.2,
            font_thickness + 2,
        )
        
        x = (width - text_width) // 2
        y = height - 100
        
        # Draw with prominent background
        self._draw_text_with_background(
            frame,
            text,
            (x, y),
            font_scale * 1.2,
            font_thickness + 2,
            self.config.text_color,
            self.config.event_color,
            padding=15,
        )
    
    def render_frame(
        self,
        frame: np.ndarray,
        frame_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        Render a single frame with overlays.
        
        Args:
            frame: Original video frame
            frame_data: Dictionary containing frame analytics
        
        Returns:
            Annotated frame
        """
        # Create a copy to avoid modifying original
        annotated = frame.copy()
        
        height, width = frame.shape[:2]
        font_scale = self._calculate_font_scale(height)
        font_thickness = self._calculate_font_thickness(height)
        
        # Extract data
        timestamp = frame_data.get("timestamp_seconds", 0.0)
        scoreboard = frame_data.get("scoreboard")
        ball = frame_data.get("ball")
        players = frame_data.get("players", [])
        events = frame_data.get("events", [])
        
        # Draw header
        self._draw_header(annotated, timestamp, font_scale, font_thickness)
        
        # Draw scoreboard
        self._draw_scoreboard(annotated, scoreboard, font_scale, font_thickness)
        
        # Draw trajectory before ball marker
        self._draw_trajectory(annotated)
        
        # Draw ball marker
        if ball:
            self._draw_ball_marker(annotated, ball, font_scale)
            
            # Update trajectory buffer
            if "x" in ball and "y" in ball:
                self.trajectory_buffer.append({"x": ball["x"], "y": ball["y"]})
                if len(self.trajectory_buffer) > self.config.trajectory_length:
                    self.trajectory_buffer.pop(0)
        
        # Draw player info
        self._draw_player_info(annotated, players, font_scale, font_thickness)
        
        # Draw events
        for event in events:
            self._draw_event_overlay(annotated, event, font_scale, font_thickness)
        
        return annotated
    
    def render_video(
        self,
        video_path: str,
        analytics_data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render full video with analytics overlays.
        
        Args:
            video_path: Path to source video
            analytics_data: Complete analytics data structure
            output_path: Optional output path
        
        Returns:
            Path to rendered video
        """
        output = output_path or self.config.output_path
        output_file = Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open source video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoRendererError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise VideoRendererError(f"Failed to create output video: {output}")
        
        # Prepare frame-indexed analytics
        frames_analytics = analytics_data.get("frames", [])
        frame_map = {f.get("frame_number", i): f for i, f in enumerate(frames_analytics)}
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get analytics for this frame
                frame_data = frame_map.get(frame_number, {})
                
                # Render overlays
                try:
                    annotated = self.render_frame(frame, frame_data)
                    out.write(annotated)
                except Exception as e:
                    # Gracefully skip problematic frames
                    print(f"Warning: Failed to render frame {frame_number}: {e}")
                    out.write(frame)
                
                frame_number += 1
        
        finally:
            cap.release()
            out.release()
        
        return str(output_file)
    
    def reset(self) -> None:
        """Reset renderer state."""
        self.trajectory_buffer.clear()


__all__ = [
    "VideoRenderer",
    "RendererConfig",
    "VideoRendererError",
]
