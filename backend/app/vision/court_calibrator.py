"""
Tennis Court Manual Calibration

Allows one-time manual keypoint selection for robust court homography.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class CourtCalibrationError(Exception):
    """Raised when court calibration fails."""


@dataclass
class CourtKeypoints:
    """Tennis court keypoints in image coordinates."""
    
    # Far baseline (top of image, far from camera)
    far_baseline_left: Tuple[float, float]
    far_baseline_right: Tuple[float, float]
    
    # Near baseline (bottom of image, near camera)
    near_baseline_left: Tuple[float, float]
    near_baseline_right: Tuple[float, float]
    
    # Service line (middle horizontal lines)
    far_service_left: Tuple[float, float]
    far_service_right: Tuple[float, float]
    near_service_left: Tuple[float, float]
    near_service_right: Tuple[float, float]
    
    # Net line
    net_left: Tuple[float, float]
    net_right: Tuple[float, float]
    
    # Center service line intersections
    far_center_t: Tuple[float, float]  # Far service line center
    near_center_t: Tuple[float, float]  # Near service line center
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "far_baseline_left": list(self.far_baseline_left),
            "far_baseline_right": list(self.far_baseline_right),
            "near_baseline_left": list(self.near_baseline_left),
            "near_baseline_right": list(self.near_baseline_right),
            "far_service_left": list(self.far_service_left),
            "far_service_right": list(self.far_service_right),
            "near_service_left": list(self.near_service_left),
            "near_service_right": list(self.near_service_right),
            "net_left": list(self.net_left),
            "net_right": list(self.net_right),
            "far_center_t": list(self.far_center_t),
            "near_center_t": list(self.near_center_t),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CourtKeypoints:
        """Load from dictionary."""
        return cls(
            far_baseline_left=tuple(data["far_baseline_left"]),
            far_baseline_right=tuple(data["far_baseline_right"]),
            near_baseline_left=tuple(data["near_baseline_left"]),
            near_baseline_right=tuple(data["near_baseline_right"]),
            far_service_left=tuple(data["far_service_left"]),
            far_service_right=tuple(data["far_service_right"]),
            near_service_left=tuple(data["near_service_left"]),
            near_service_right=tuple(data["near_service_right"]),
            net_left=tuple(data["net_left"]),
            net_right=tuple(data["net_right"]),
            far_center_t=tuple(data["far_center_t"]),
            near_center_t=tuple(data["near_center_t"]),
        )


class TennisCourtModel:
    """
    Standard tennis court dimensions (singles).
    
    Origin at center of near baseline.
    X-axis: left (-) to right (+)
    Y-axis: near (0) to far (+)
    
    All measurements in meters.
    """
    
    # Court dimensions (singles)
    COURT_LENGTH = 23.77  # baseline to baseline
    COURT_WIDTH = 8.23  # singles sideline to sideline
    
    # Service box
    SERVICE_LINE_DISTANCE = 6.40  # from net
    NET_DISTANCE_FROM_BASELINE = 11.885  # half court
    
    # Center service line
    CENTER_SERVICE_WIDTH = COURT_WIDTH / 2.0
    
    @classmethod
    def get_canonical_keypoints(cls) -> Dict[str, Tuple[float, float]]:
        """
        Return canonical court keypoints in real-world coordinates.
        
        Returns keypoints matching CourtKeypoints structure.
        """
        half_width = cls.COURT_WIDTH / 2.0
        
        return {
            # Far baseline (y = COURT_LENGTH)
            "far_baseline_left": (-half_width, cls.COURT_LENGTH),
            "far_baseline_right": (half_width, cls.COURT_LENGTH),
            
            # Near baseline (y = 0)
            "near_baseline_left": (-half_width, 0.0),
            "near_baseline_right": (half_width, 0.0),
            
            # Far service line (y = NET + SERVICE_LINE_DISTANCE)
            "far_service_left": (
                -half_width,
                cls.NET_DISTANCE_FROM_BASELINE + cls.SERVICE_LINE_DISTANCE
            ),
            "far_service_right": (
                half_width,
                cls.NET_DISTANCE_FROM_BASELINE + cls.SERVICE_LINE_DISTANCE
            ),
            
            # Near service line (y = NET - SERVICE_LINE_DISTANCE)
            "near_service_left": (
                -half_width,
                cls.NET_DISTANCE_FROM_BASELINE - cls.SERVICE_LINE_DISTANCE
            ),
            "near_service_right": (
                half_width,
                cls.NET_DISTANCE_FROM_BASELINE - cls.SERVICE_LINE_DISTANCE
            ),
            
            # Net line (y = NET_DISTANCE_FROM_BASELINE)
            "net_left": (-half_width, cls.NET_DISTANCE_FROM_BASELINE),
            "net_right": (half_width, cls.NET_DISTANCE_FROM_BASELINE),
            
            # Center service line T-intersections
            "far_center_t": (
                0.0,
                cls.NET_DISTANCE_FROM_BASELINE + cls.SERVICE_LINE_DISTANCE
            ),
            "near_center_t": (
                0.0,
                cls.NET_DISTANCE_FROM_BASELINE - cls.SERVICE_LINE_DISTANCE
            ),
        }


class CourtCalibrator:
    """
    Manual tennis court calibration tool.
    
    Usage:
        1. Extract a clear frame from video
        2. Run interactive calibration tool
        3. Click court keypoints in order
        4. Save calibration file
        5. Load calibration for processing
    """
    
    def __init__(self):
        self.keypoints: Optional[CourtKeypoints] = None
        self.homography_matrix: Optional[np.ndarray] = None
        self.court_model = TennisCourtModel()
    
    def compute_homography(self, keypoints: CourtKeypoints) -> np.ndarray:
        """
        Compute homography from image keypoints to canonical court model.
        
        Uses 4-point correspondence for perspective transform.
        """
        # Get canonical court coordinates
        canonical = self.court_model.get_canonical_keypoints()
        
        # Use 4 main corners for homography (most reliable)
        src_points = np.array([
            keypoints.near_baseline_left,
            keypoints.near_baseline_right,
            keypoints.far_baseline_right,
            keypoints.far_baseline_left,
        ], dtype=np.float32)
        
        dst_points = np.array([
            canonical["near_baseline_left"],
            canonical["near_baseline_right"],
            canonical["far_baseline_right"],
            canonical["far_baseline_left"],
        ], dtype=np.float32)
        
        # Compute homography matrix
        H, status = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
        
        if H is None:
            raise CourtCalibrationError("Failed to compute homography matrix")
        
        return H
    
    def calibrate_from_keypoints(self, keypoints: CourtKeypoints) -> None:
        """Set calibration from keypoints."""
        self.keypoints = keypoints
        self.homography_matrix = self.compute_homography(keypoints)
    
    def save_calibration(self, path: str) -> None:
        """Save calibration to JSON file."""
        if self.keypoints is None or self.homography_matrix is None:
            raise CourtCalibrationError("No calibration to save")
        
        data = {
            "keypoints": self.keypoints.to_dict(),
            "homography_matrix": self.homography_matrix.tolist(),
        }
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_calibration(self, path: str) -> None:
        """Load calibration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.keypoints = CourtKeypoints.from_dict(data["keypoints"])
        self.homography_matrix = np.array(data["homography_matrix"], dtype=np.float32)
    
    def image_to_court_coordinates(
        self,
        image_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transform image pixel coordinates to court coordinates (meters).
        
        Returns None if transformation fails.
        """
        if self.homography_matrix is None:
            return None
        
        # Convert to homogeneous coordinates
        pt = np.array([[image_point[0], image_point[1]]], dtype=np.float32)
        pt = pt.reshape(-1, 1, 2)
        
        # Apply homography
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        
        if transformed is None or transformed.size == 0:
            return None
        
        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))
    
    def court_to_image_coordinates(
        self,
        court_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transform court coordinates (meters) to image pixel coordinates.
        
        Returns None if transformation fails.
        """
        if self.homography_matrix is None:
            return None
        
        # Invert homography
        H_inv = np.linalg.inv(self.homography_matrix)
        
        # Convert to homogeneous coordinates
        pt = np.array([[court_point[0], court_point[1]]], dtype=np.float32)
        pt = pt.reshape(-1, 1, 2)
        
        # Apply inverse homography
        transformed = cv2.perspectiveTransform(pt, H_inv)
        
        if transformed is None or transformed.size == 0:
            return None
        
        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))
    
    def draw_court_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw calibrated court overlay on frame.
        
        Shows all court lines in canonical positions.
        """
        if self.keypoints is None:
            return frame
        
        overlay = frame.copy()
        kp = self.keypoints
        
        # Define court lines as point pairs
        lines = [
            # Baselines
            (kp.far_baseline_left, kp.far_baseline_right),
            (kp.near_baseline_left, kp.near_baseline_right),
            
            # Sidelines
            (kp.far_baseline_left, kp.near_baseline_left),
            (kp.far_baseline_right, kp.near_baseline_right),
            
            # Service lines
            (kp.far_service_left, kp.far_service_right),
            (kp.near_service_left, kp.near_service_right),
            
            # Net
            (kp.net_left, kp.net_right),
            
            # Center service line
            (kp.far_center_t, kp.near_center_t),
        ]
        
        # Draw lines
        for pt1, pt2 in lines:
            cv2.line(
                overlay,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                (0, 255, 255),  # Yellow
                2,
                cv2.LINE_AA
            )
        
        # Draw keypoints
        for pt in [
            kp.far_baseline_left, kp.far_baseline_right,
            kp.near_baseline_left, kp.near_baseline_right,
            kp.far_service_left, kp.far_service_right,
            kp.near_service_left, kp.near_service_right,
            kp.net_left, kp.net_right,
            kp.far_center_t, kp.near_center_t,
        ]:
            cv2.circle(
                overlay,
                (int(pt[0]), int(pt[1])),
                5,
                (0, 255, 255),  # Yellow
                -1
            )
        
        return overlay
    
    def is_point_in_court(
        self,
        image_point: Tuple[float, float],
        margin_meters: float = 2.0
    ) -> bool:
        """
        Check if image point falls within court boundaries (with margin).
        
        Returns True if point is inside court + margin.
        """
        court_point = self.image_to_court_coordinates(image_point)
        if court_point is None:
            return False
        
        x, y = court_point
        
        half_width = self.court_model.COURT_WIDTH / 2.0
        
        # Check bounds with margin
        in_x = (-half_width - margin_meters) <= x <= (half_width + margin_meters)
        in_y = (0.0 - margin_meters) <= y <= (self.court_model.COURT_LENGTH + margin_meters)
        
        return in_x and in_y
    
    def get_court_side(
        self,
        image_point: Tuple[float, float]
    ) -> Optional[str]:
        """
        Determine which side of net the point is on.
        
        Returns "near", "far", or None if transformation fails.
        """
        court_point = self.image_to_court_coordinates(image_point)
        if court_point is None:
            return None
        
        _, y = court_point
        net_y = self.court_model.NET_DISTANCE_FROM_BASELINE
        
        return "near" if y < net_y else "far"
