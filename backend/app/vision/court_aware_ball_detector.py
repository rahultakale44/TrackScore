"""
Court-aware tennis ball detection with temporal filtering.

Improves on generic YOLO ball detection by:
- Restricting search to calibrated court ROI
- Using trajectory prediction
- Rejecting outliers
- Applying Kalman filtering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ball_detector import BallDetector, BallDetectorConfig
from .court_calibrator import CourtCalibrator


class CourtAwareBallDetectorError(Exception):
    """Raised when court-aware ball detection fails."""


@dataclass
class CourtAwareBallConfig:
    """Configuration for court-aware ball detection."""
    
    # YOLO detection
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.03
    
    # Court ROI
    court_margin_meters: float = 5.0  # Search outside court boundaries
    
    # Temporal filtering
    use_kalman: bool = True
    max_velocity_pixels_per_second: float = 2000.0
    max_acceleration_pixels_per_second2: float = 5000.0
    
    # Trajectory consistency
    max_distance_from_prediction_pixels: float = 150.0
    
    # History
    trajectory_length: int = 60
    
    device: Optional[str] = None


class SimpleKalmanFilter:
    """
    Simple 2D Kalman filter for ball position tracking.
    
    State: [x, y, vx, vy]
    """
    
    def __init__(self):
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)
        
        # State covariance
        self.P = np.eye(4, dtype=np.float32) * 1000
        
        # Process noise
        self.Q = np.eye(4, dtype=np.float32)
        self.Q[0:2, 0:2] *= 10  # Position noise
        self.Q[2:4, 2:4] *= 100  # Velocity noise
        
        # Measurement noise
        self.R = np.eye(2, dtype=np.float32) * 25
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.initialized = False
    
    def predict(self, dt: float) -> Tuple[float, float]:
        """
        Predict next state.
        
        Returns predicted (x, y) position.
        """
        if not self.initialized:
            return (0.0, 0.0)
        
        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Predict
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        return (float(self.state[0, 0]), float(self.state[1, 0]))
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update filter with measurement.
        
        Returns filtered (x, y) position.
        """
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        
        if not self.initialized:
            # Initialize state with first measurement
            self.state[0, 0] = measurement[0]
            self.state[1, 0] = measurement[1]
            self.state[2, 0] = 0.0
            self.state[3, 0] = 0.0
            self.initialized = True
            return measurement
        
        # Innovation
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return (float(self.state[0, 0]), float(self.state[1, 0]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (float(self.state[2, 0]), float(self.state[3, 0]))
    
    def reset(self):
        """Reset filter state."""
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.initialized = False


class CourtAwareBallDetector:
    """
    Detect tennis ball using court context and temporal filtering.
    """
    
    def __init__(
        self,
        court_calibrator: CourtCalibrator,
        config: CourtAwareBallConfig | None = None
    ):
        if court_calibrator.homography_matrix is None:
            raise CourtAwareBallDetectorError(
                "Court calibrator must have valid homography"
            )
        
        self.calibrator = court_calibrator
        self.config = config if config is not None else CourtAwareBallConfig()
        
        # Initialize YOLO detector
        detector_config = BallDetectorConfig(
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
        )
        self.detector = BallDetector(detector_config)
        
        # Kalman filter
        self.kalman = SimpleKalmanFilter() if self.config.use_kalman else None
        
        # Trajectory history
        self.trajectory: List[Dict[str, Any]] = []
        
        # Last known state
        self.last_position: Optional[Tuple[float, float]] = None
        self.last_timestamp: Optional[float] = None
        self.missed_frames: int = 0
    
    def _create_court_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create ROI mask covering court area with margin.
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if self.calibrator.keypoints is None:
            # No calibration - use full frame
            return np.ones((height, width), dtype=np.uint8) * 255
        
        # Get court corners and expand
        kp = self.calibrator.keypoints
        
        # Get all court boundary points
        court_points = [
            kp.far_baseline_left,
            kp.far_baseline_right,
            kp.near_baseline_right,
            kp.near_baseline_left,
        ]
        
        # Convert to numpy array
        poly = np.array([court_points], dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(mask, poly, 255)
        
        # Dilate to add margin
        kernel_size = int(width * 0.05)  # 5% margin
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def detect_ball(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float
    ) -> Dict[str, Any]:
        """
        Detect tennis ball with court awareness and temporal filtering.
        
        Returns:
            {
                "frame_number": int,
                "timestamp_seconds": float,
                "all_candidates": [...],  # All YOLO detections
                "roi_candidates": [...],  # Candidates inside court ROI
                "filtered_candidates": [...],  # After temporal filtering
                "selected_ball": {...} or None,
                "predicted_ball": {...} or None,
                "ball_visible": bool,
                "trajectory": [...]
            }
        """
        # Create court ROI mask
        roi_mask = self._create_court_roi_mask(frame)
        
        # Predict next position if we have history
        dt = 0.0333  # Assume 30 fps
        if self.last_timestamp is not None:
            dt = timestamp_seconds - self.last_timestamp
        
        predicted_position = None
        if self.kalman and self.kalman.initialized:
            predicted_position = self.kalman.predict(dt)
        elif self.last_position is not None and self.last_timestamp is not None:
            # Simple linear prediction
            if len(self.trajectory) >= 2:
                prev = self.trajectory[-1]
                prev_prev = self.trajectory[-2]
                vx = (prev["x"] - prev_prev["x"]) / (prev["timestamp_seconds"] - prev_prev["timestamp_seconds"])
                vy = (prev["y"] - prev_prev["y"]) / (prev["timestamp_seconds"] - prev_prev["timestamp_seconds"])
                predicted_position = (
                    self.last_position[0] + vx * dt,
                    self.last_position[1] + vy * dt
                )
        
        # Run YOLO detection
        yolo_result = self.detector.detect_ball(frame)
        all_candidates = yolo_result.get("candidates", [])
        
        # Filter by ROI
        roi_candidates = []
        for candidate in all_candidates:
            center = candidate.get("center", {})
            cx, cy = center.get("x", 0), center.get("y", 0)
            
            # Check if inside ROI mask
            if 0 <= int(cy) < roi_mask.shape[0] and 0 <= int(cx) < roi_mask.shape[1]:
                if roi_mask[int(cy), int(cx)] > 0:
                    roi_candidates.append(candidate)
        
        # Filter by temporal consistency
        filtered_candidates = []
        if predicted_position is not None:
            for candidate in roi_candidates:
                center = candidate.get("center", {})
                cx, cy = center.get("x", 0), center.get("y", 0)
                
                # Distance from prediction
                dist = np.hypot(
                    cx - predicted_position[0],
                    cy - predicted_position[1]
                )
                
                if dist < self.config.max_distance_from_prediction_pixels:
                    candidate["distance_from_prediction"] = float(dist)
                    filtered_candidates.append(candidate)
        else:
            # No prediction - use all ROI candidates
            filtered_candidates = roi_candidates
        
        # Select best candidate
        selected_ball = None
        if filtered_candidates:
            # Sort by confidence * (1 - normalized_distance)
            for candidate in filtered_candidates:
                conf = candidate.get("confidence", 0.0)
                dist = candidate.get("distance_from_prediction", 0.0)
                dist_normalized = min(1.0, dist / self.config.max_distance_from_prediction_pixels)
                candidate["selection_score"] = conf * (1.0 - dist_normalized * 0.3)
            
            filtered_candidates.sort(key=lambda c: c.get("selection_score", 0), reverse=True)
            selected_ball = filtered_candidates[0]
            
            # Update Kalman filter
            center = selected_ball.get("center", {})
            measurement = (float(center.get("x", 0)), float(center.get("y", 0)))
            
            if self.kalman:
                filtered_pos = self.kalman.update(measurement)
                selected_ball["filtered_position"] = {
                    "x": filtered_pos[0],
                    "y": filtered_pos[1]
                }
                velocity = self.kalman.get_velocity()
                selected_ball["velocity"] = {
                    "x": velocity[0],
                    "y": velocity[1]
                }
            else:
                selected_ball["filtered_position"] = {
                    "x": measurement[0],
                    "y": measurement[1]
                }
            
            # Update state
            self.last_position = (
                selected_ball["filtered_position"]["x"],
                selected_ball["filtered_position"]["y"]
            )
            self.last_timestamp = timestamp_seconds
            self.missed_frames = 0
            
            # Add to trajectory
            self.trajectory.append({
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "x": self.last_position[0],
                "y": self.last_position[1],
                "predicted": False,
                "confidence": selected_ball.get("confidence", 0.0)
            })
        
        else:
            # No detection - use prediction if available
            self.missed_frames += 1
            
            if predicted_position is not None and self.missed_frames < 10:
                selected_ball = {
                    "filtered_position": {
                        "x": predicted_position[0],
                        "y": predicted_position[1]
                    },
                    "predicted": True,
                    "confidence": 0.0
                }
                
                self.last_position = predicted_position
                self.last_timestamp = timestamp_seconds
                
                # Add predicted point to trajectory
                self.trajectory.append({
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "x": predicted_position[0],
                    "y": predicted_position[1],
                    "predicted": True,
                    "confidence": 0.0
                })
            else:
                # Lost track
                self.last_position = None
                if self.kalman:
                    self.kalman.reset()
        
        # Trim trajectory
        if len(self.trajectory) > self.config.trajectory_length:
            self.trajectory = self.trajectory[-self.config.trajectory_length:]
        
        return {
            "frame_number": frame_number,
            "timestamp_seconds": timestamp_seconds,
            "all_candidates": all_candidates,
            "roi_candidates": roi_candidates,
            "filtered_candidates": filtered_candidates,
            "selected_ball": selected_ball,
            "predicted_position": predicted_position,
            "ball_visible": selected_ball is not None and not selected_ball.get("predicted", False),
            "trajectory": list(self.trajectory),
        }
    
    def reset(self):
        """Reset detection state."""
        if self.kalman:
            self.kalman.reset()
        self.trajectory.clear()
        self.last_position = None
        self.last_timestamp = None
        self.missed_frames = 0
