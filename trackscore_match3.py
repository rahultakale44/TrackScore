#!/usr/bin/env python3
"""
TrackScore Core Perception Pipeline for tennis_match3.mp4

Uses calibrated court geometry for robust player/ball detection.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Vision modules
from backend.app.vision.video_loader import VideoLoader
from backend.app.vision.player_detector import PlayerDetector
from backend.app.vision.player_tracker import PlayerTracker
from backend.app.vision.ball_detector import BallDetector
from backend.app.vision.robust_player_filter import RobustPlayerFilter


class CourtCalibration:
    """Load and use court calibration."""
    
    def __init__(self, calib_path: str):
        with open(calib_path, 'r') as f:
            data = json.load(f)
        
        self.homography = np.array(data["homography_matrix"], dtype=np.float32)
        self.homography_inv = np.linalg.inv(self.homography)
        self.keypoints = data["keypoints"]
        self.reprojection_error = data["reprojection_error_meters"]
        
        print(f"✓ Court calibration loaded: {calib_path}")
        print(f"  Reprojection error: {self.reprojection_error:.3f}m")
    
    def image_to_court(self, image_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transform image pixel to court coordinates (meters)."""
        pt = np.array([[image_point[0], image_point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        court_pt = cv2.perspectiveTransform(pt, self.homography)
        return (float(court_pt[0, 0, 0]), float(court_pt[0, 0, 1]))
    
    def court_to_image(self, court_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transform court coordinates to image pixels."""
        pt = np.array([[court_point[0], court_point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        img_pt = cv2.perspectiveTransform(pt, self.homography_inv)
        return (float(img_pt[0, 0, 0]), float(img_pt[0, 0, 1]))
    
    def draw_court_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw projected court lines on frame."""
        overlay = frame.copy()
        
        # Define court lines in court coordinates
        lines = [
            # Far baseline
            ((-4.115, 23.77), (4.115, 23.77)),
            # Near baseline  
            ((-4.115, 0.0), (4.115, 0.0)),
            # Left sideline
            ((-4.115, 0.0), (-4.115, 23.77)),
            # Right sideline
            ((4.115, 0.0), (4.115, 23.77)),
            # Far service line
            ((-4.115, 18.285), (4.115, 18.285)),
            # Near service line
            ((-4.115, 5.485), (4.115, 5.485)),
            # Net
            ((-4.115, 11.885), (4.115, 11.885)),
            # Center service line
            ((0.0, 5.485), (0.0, 18.285)),
        ]
        
        for pt1, pt2 in lines:
            try:
                img_pt1 = self.court_to_image(pt1)
                img_pt2 = self.court_to_image(pt2)
                cv2.line(overlay,
                        (int(img_pt1[0]), int(img_pt1[1])),
                        (int(img_pt2[0]), int(img_pt2[1])),
                        (0, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
        
        return overlay


class TennisBallDetector:
    """Tennis-specific ball detection with strict trajectory validation."""
    
    def __init__(self, court_calib: CourtCalibration):
        self.court = court_calib
        self.ball_detector = BallDetector()
        
        # Separate histories for confirmed vs predicted
        self.confirmed_trajectory: List[Dict] = []
        self.predicted_history: List[Dict] = []
        
        # Kalman filter for short-term prediction only
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state (x,y,vx,vy), 2 measurement (x,y)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
        
        self.last_confirmed_pos: Optional[Tuple[float, float]] = None
        self.last_confirmed_velocity: Optional[Tuple[float, float]] = None
        self.consecutive_predictions = 0
        self.initialized = False
        self.trajectory_segment_id = 0
    
    def create_court_roi_mask(self, frame_shape) -> np.ndarray:
        """Create ROI mask for ball detection (court + margin)."""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get court boundary points with margin
        margin = 2.0  # meters
        court_polygon = [
            (-4.115 - margin, -margin),
            (4.115 + margin, -margin),
            (4.115 + margin, 23.77 + margin),
            (-4.115 - margin, 23.77 + margin),
        ]
        
        # Project to image space
        img_points = []
        for pt in court_polygon:
            img_pt = self.court.court_to_image(pt)
            img_points.append([int(img_pt[0]), int(img_pt[1])])
        
        cv2.fillPoly(mask, [np.array(img_points)], 255)
        return mask
    
    def _detect_tiny_bright_candidates(self, frame: np.ndarray, roi_mask: np.ndarray) -> List[Dict]:
        """Detect small bright circular objects using frame differencing and contours."""
        candidates = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply court ROI mask
        gray_masked = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # Threshold for bright objects
        _, bright_mask = cv2.threshold(gray_masked, 180, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Tennis ball is small: typically 5-30 pixels diameter at this resolution
            if area < 10 or area > 500:
                continue
            
            # Get bounding circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < 0.5:  # Not circular enough
                continue
            
            # Calculate appearance score
            x1, y1 = int(cx - radius), int(cy - radius)
            x2, y2 = int(cx + radius), int(cy + radius)
            
            # Bounds check
            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Check for tennis ball yellow/green color
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv_crop, np.array([20, 40, 80]), np.array([45, 255, 255]))
            yellow_ratio = np.count_nonzero(yellow_mask) / yellow_mask.size if yellow_mask.size > 0 else 0
            
            candidates.append({
                "center": {"x": float(cx), "y": float(cy)},
                "radius": float(radius),
                "area": float(area),
                "circularity": float(circularity),
                "yellow_ratio": float(yellow_ratio),
                "appearance_score": float(circularity * 0.4 + yellow_ratio * 0.6),
                "source": "tiny_bright_contour"
            })
        
        return candidates
    
    def _is_static_object(self, x: float, y: float, prev_frame_gray: Optional[np.ndarray], 
                          curr_frame_gray: np.ndarray) -> bool:
        """Check if position corresponds to a static object (court line, scoreboard, etc)."""
        if prev_frame_gray is None:
            return False
        
        # Check frame difference at this location
        radius = 10
        x1, y1 = max(0, int(x - radius)), max(0, int(y - radius))
        x2, y2 = min(curr_frame_gray.shape[1], int(x + radius)), min(curr_frame_gray.shape[0], int(y + radius))
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        prev_crop = prev_frame_gray[y1:y2, x1:x2]
        curr_crop = curr_frame_gray[y1:y2, x1:x2]
        
        if prev_crop.size == 0 or curr_crop.size == 0:
            return False
        
        # Calculate difference
        diff = cv2.absdiff(prev_crop, curr_crop)
        mean_diff = np.mean(diff)
        
        # If very little change, likely static
        return mean_diff < 5.0
    
    def _score_candidate(self, candidate: Dict, predicted_pos: Optional[Tuple[float, float]],
                        prev_frame_gray: Optional[np.ndarray], curr_frame_gray: np.ndarray) -> float:
        """Score a ball candidate using multiple signals."""
        cx = candidate["center"]["x"]
        cy = candidate["center"]["y"]
        
        score = 0.0
        
        # 1. Detector confidence (if available)
        detector_conf = candidate.get("confidence", candidate.get("appearance_score", 0.0))
        score += detector_conf * 0.25
        
        # 2. Appearance score (circularity, color)
        appearance = candidate.get("appearance_score", candidate.get("circularity", 0.5))
        score += appearance * 0.25
        
        # 3. Trajectory consistency (distance from prediction)
        if predicted_pos:
            dist = np.hypot(cx - predicted_pos[0], cy - predicted_pos[1])
            if dist < 80:  # Reasonable range
                proximity_score = (1.0 - dist / 80.0)
                score += proximity_score * 0.35
            else:
                # Large jump penalty
                score -= 0.3
        else:
            # No prediction, no penalty
            score += 0.15
        
        # 4. Motion score (not static)
        if self._is_static_object(cx, cy, prev_frame_gray, curr_frame_gray):
            score -= 0.4  # Heavy penalty for static objects
        else:
            score += 0.15
        
        return score
    
    def _check_plausible_reacquisition(self, new_pos: Tuple[float, float], 
                                       time_gap: float) -> bool:
        """Check if a new detection can plausibly reconnect to the previous trajectory."""
        if not self.last_confirmed_pos or time_gap > 0.5:  # More than 0.5s gap
            return False
        
        # Check spatial gap
        dist = np.hypot(new_pos[0] - self.last_confirmed_pos[0], 
                       new_pos[1] - self.last_confirmed_pos[1])
        
        # Tennis ball max speed ~300 km/h = ~83 m/s
        # In image space, check against velocity-based threshold
        if self.last_confirmed_velocity:
            expected_dist = np.hypot(self.last_confirmed_velocity[0], 
                                    self.last_confirmed_velocity[1]) * time_gap
            # Allow 3x expected distance as tolerance
            if dist > expected_dist * 3 and dist > 200:
                return False
        else:
            # No velocity info, use distance only
            if dist > 300:  # Large spatial gap
                return False
        
        return True
    
    def detect_ball(self, frame: np.ndarray, frame_number: int, timestamp: float,
                   prev_frame_gray: Optional[np.ndarray] = None) -> Dict:
        """
        Detect tennis ball with strict trajectory validation.
        
        Returns confirmed detection or short-term prediction (max 3 frames).
        """
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get all candidate sources
        all_candidates = []
        
        # 1. YOLO ball detection
        roi_mask = self.create_court_roi_mask(frame.shape)
        result = self.ball_detector.detect_ball(frame)
        yolo_candidates = result.get("candidates", [])
        
        # Filter by ROI
        for cand in yolo_candidates:
            center = cand.get("center", {})
            cx, cy = int(center.get("x", 0)), int(center.get("y", 0))
            if 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]:
                if roi_mask[cy, cx] > 0:
                    cand["source"] = "yolo"
                    all_candidates.append(cand)
        
        # 2. Tiny bright contour candidates
        contour_candidates = self._detect_tiny_bright_candidates(frame, roi_mask)
        all_candidates.extend(contour_candidates)
        
        # 3. High-resolution crop around predicted position
        predicted_pos = None
        if self.initialized and self.last_confirmed_pos:
            prediction = self.kf.predict()
            predicted_pos = (float(prediction[0, 0]), float(prediction[1, 0]))
            
            search_radius = 100
            lx, ly = int(predicted_pos[0]), int(predicted_pos[1])
            x1 = max(0, lx - search_radius)
            y1 = max(0, ly - search_radius)
            x2 = min(frame.shape[1], lx + search_radius)
            y2 = min(frame.shape[0], ly + search_radius)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crop_result = self.ball_detector.detect_ball(crop)
                crop_candidates = crop_result.get("candidates", [])
                
                for cand in crop_candidates:
                    center = cand.get("center", {})
                    center["x"] = center.get("x", 0) + x1
                    center["y"] = center.get("y", 0) + y1
                    cand["source"] = "high_res_crop"
                    
                    cx, cy = int(center["x"]), int(center["y"])
                    if roi_mask[cy, cx] > 0:
                        all_candidates.append(cand)
        
        # Score all candidates
        scored_candidates = []
        for cand in all_candidates:
            score = self._score_candidate(cand, predicted_pos, prev_frame_gray, curr_frame_gray)
            scored_candidates.append((score, cand))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select best candidate if score is good enough
        selected_ball = None
        MIN_SCORE_THRESHOLD = 0.3
        
        if scored_candidates and scored_candidates[0][0] >= MIN_SCORE_THRESHOLD:
            selected_ball = scored_candidates[0][1]
            center = selected_ball.get("center", {})
            cx, cy = center.get("x", 0), center.get("y", 0)
            
            # Check if this is a plausible reacquisition after loss
            if self.consecutive_predictions > 0:
                time_gap = timestamp - (self.confirmed_trajectory[-1]["timestamp"] if self.confirmed_trajectory else 0)
                if not self._check_plausible_reacquisition((cx, cy), time_gap):
                    # Start new trajectory segment
                    self.trajectory_segment_id += 1
                    self.initialized = False
                    self.predicted_history.clear()
            
            # Update Kalman filter
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            
            if not self.initialized:
                self.kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                self.initialized = True
            
            self.kf.correct(measurement)
            state = self.kf.statePost
            
            # Calculate velocity
            velocity = (float(state[2, 0]), float(state[3, 0]))
            
            # Add to CONFIRMED trajectory
            self.confirmed_trajectory.append({
                "frame": frame_number,
                "timestamp": timestamp,
                "x": cx,
                "y": cy,
                "predicted": False,
                "confidence": selected_ball.get("confidence", scored_candidates[0][0]),
                "segment_id": self.trajectory_segment_id,
                "source": selected_ball.get("source", "unknown")
            })
            
            self.last_confirmed_pos = (cx, cy)
            self.last_confirmed_velocity = velocity
            self.consecutive_predictions = 0
            self.predicted_history.clear()
            
            return {
                "all_candidates": all_candidates,
                "scored_candidates": [(s, c.get("source", "?")) for s, c in scored_candidates[:5]],
                "ball": {
                    "x": cx,
                    "y": cy,
                    "predicted": False,
                    "confidence": selected_ball.get("confidence", scored_candidates[0][0]),
                    "source": selected_ball.get("source", "unknown")
                },
                "confirmed_trajectory": list(self.confirmed_trajectory[-60:]),
                "predicted_history": [],
                "status": "DETECTED"
            }
        
        # No good detection - use prediction for MAX 3 frames
        MAX_CONSECUTIVE_PREDICTIONS = 3
        
        if predicted_pos and self.consecutive_predictions < MAX_CONSECUTIVE_PREDICTIONS:
            self.consecutive_predictions += 1
            
            # Add to PREDICTED history (separate from confirmed)
            self.predicted_history.append({
                "frame": frame_number,
                "timestamp": timestamp,
                "x": predicted_pos[0],
                "y": predicted_pos[1],
                "predicted": True,
                "confidence": 0.0
            })
            
            return {
                "all_candidates": all_candidates,
                "scored_candidates": [(s, c.get("source", "?")) for s, c in scored_candidates[:5]],
                "ball": {
                    "x": predicted_pos[0],
                    "y": predicted_pos[1],
                    "predicted": True,
                    "confidence": 0.0,
                    "prediction_count": self.consecutive_predictions
                },
                "confirmed_trajectory": list(self.confirmed_trajectory[-60:]),
                "predicted_history": list(self.predicted_history),
                "status": f"PREDICTED_{self.consecutive_predictions}"
            }
        
        # Ball LOST - clear predictions and wait for reacquisition
        self.consecutive_predictions += 1
        self.predicted_history.clear()
        
        if self.consecutive_predictions > 10:
            # Too many lost frames, reset tracking
            self.initialized = False
            self.trajectory_segment_id += 1
        
        return {
            "all_candidates": all_candidates,
            "scored_candidates": [(s, c.get("source", "?")) for s, c in scored_candidates[:5]],
            "ball": None,
            "confirmed_trajectory": list(self.confirmed_trajectory[-60:]),
            "predicted_history": [],
            "status": "LOST"
        }


def calculate_player_speed(
    court_calib: CourtCalibration,
    player_history: List[Dict],
    min_samples: int = 5
) -> Optional[float]:
    """Calculate player speed in km/h from court-projected positions."""
    if len(player_history) < min_samples:
        return None
    
    recent = player_history[-min_samples:]
    total_dist = 0.0
    total_time = 0.0
    
    for i in range(1, len(recent)):
        prev = recent[i-1]
        curr = recent[i]
        
        # Get court coordinates
        prev_court = court_calib.image_to_court(prev["foot_pos"])
        curr_court = court_calib.image_to_court(curr["foot_pos"])
        
        if prev_court and curr_court:
            dist = np.hypot(curr_court[0] - prev_court[0], curr_court[1] - prev_court[1])
            dt = curr["timestamp"] - prev["timestamp"]
            
            if dt > 0 and dist < 5.0:  # sanity check: less than 5m in one frame
                total_dist += dist
                total_time += dt
    
    if total_time > 0:
        speed_mps = total_dist / total_time
        speed_kmh = speed_mps * 3.6
        return speed_kmh if speed_kmh < 40 else None  # max plausible: 40 km/h
    
    return None


def calculate_ball_speed(
    court_calib: CourtCalibration,
    confirmed_trajectory: List[Dict],
    min_samples: int = 3
) -> Optional[float]:
    """
    Calculate ball speed in km/h using ONLY confirmed detections with calibrated court coordinates.
    
    Never uses predicted positions.
    """
    if len(confirmed_trajectory) < min_samples:
        return None
    
    # Use only confirmed (non-predicted) recent detections
    recent_confirmed = [t for t in confirmed_trajectory[-15:] if not t.get("predicted", False)]
    
    if len(recent_confirmed) < min_samples:
        return None
    
    speeds = []
    for i in range(1, len(recent_confirmed)):
        prev, curr = recent_confirmed[i-1], recent_confirmed[i]
        
        # Transform to court coordinates
        try:
            prev_court = court_calib.image_to_court((prev["x"], prev["y"]))
            curr_court = court_calib.image_to_court((curr["x"], curr["y"]))
            
            if prev_court and curr_court:
                # Calculate distance in meters on court plane
                dist_m = np.hypot(
                    curr_court[0] - prev_court[0],
                    curr_court[1] - prev_court[1]
                )
                dt = curr["timestamp"] - prev["timestamp"]
                
                # Sanity checks
                if dt > 0 and dist_m < 15.0:  # Ball doesn't move >15m in one frame
                    speed_mps = dist_m / dt
                    speed_kmh = speed_mps * 3.6
                    
                    # Plausible tennis ball speed range
                    if 10 < speed_kmh < 300:
                        # Additional confidence check
                        min_conf = min(prev.get("confidence", 0), curr.get("confidence", 0))
                        if min_conf > 0.2:  # Both detections must be confident
                            speeds.append(speed_kmh)
        except:
            # Skip if transformation fails
            continue
    
    if len(speeds) >= 2:
        # Use median to reject outliers
        return float(np.median(speeds))
    
    return None


def draw_debug_frame(
    frame: np.ndarray,
    court: CourtCalibration,
    filter_result: Dict,
    ball_result: Dict
) -> np.ndarray:
    """Create debug visualization."""
    debug = frame.copy()
    
    # Draw court overlay
    debug = court.draw_court_overlay(debug)
    
    # Draw all person detections in yellow
    for person in filter_result.get("all_persons", []):
        bbox = person.get("bbox", {})
        if isinstance(bbox, dict):
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 1)
    
    # Draw rejected persons in red
    for person in filter_result.get("rejected_persons", []):
        bbox = person.get("bbox", {})
        reason = person.get("rejection_reason", "")
        if isinstance(bbox, dict):
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(debug, f"REJECT: {reason}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Draw Player A in green
    player_a = filter_result.get("player_a")
    if player_a:
        bbox = player_a.get("bbox", {})
        if isinstance(bbox, dict):
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(debug, "PLAYER A", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw Player B in green
    player_b = filter_result.get("player_b")
    if player_b:
        bbox = player_b.get("bbox", {})
        if isinstance(bbox, dict):
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(debug, "PLAYER B", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw ball candidates (orange - all candidates)
    for cand in ball_result.get("all_candidates", [])[:20]:  # Limit display
        center = cand.get("center", {})
        cx, cy = int(center.get("x", 0)), int(center.get("y", 0))
        cv2.circle(debug, (cx, cy), 3, (0, 165, 255), 1)
    
    # Draw selected ball
    ball = ball_result.get("ball")
    if ball:
        bx, by = int(ball["x"]), int(ball["y"])
        is_pred = ball.get("predicted", False)
        pred_count = ball.get("prediction_count", 0)
        
        if is_pred:
            # Blue for predictions, lighter as count increases
            intensity = max(100, 255 - pred_count * 50)
            color = (intensity, 0, 0)
        else:
            # Green for confirmed detections
            color = (0, 255, 0)
        
        cv2.circle(debug, (bx, by), 10, color, 2)
        cv2.circle(debug, (bx, by), 3, color, -1)
        
        # Show status
        status = ball_result.get("status", "")
        cv2.putText(debug, status, (bx + 15, by),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw CONFIRMED trajectory (dim gray for history)
    confirmed_traj = ball_result.get("confirmed_trajectory", [])
    if len(confirmed_traj) >= 2:
        for i in range(1, len(confirmed_traj)):
            prev, curr = confirmed_traj[i-1], confirmed_traj[i]
            
            # Only connect within same segment
            if prev.get("segment_id") == curr.get("segment_id"):
                pt1 = (int(prev["x"]), int(prev["y"]))
                pt2 = (int(curr["x"]), int(curr["y"]))
                cv2.line(debug, pt1, pt2, (128, 128, 128), 1, cv2.LINE_AA)
    
    # Draw SHORT-TERM predictions (blue, dashed)
    predicted_hist = ball_result.get("predicted_history", [])
    if len(predicted_hist) >= 2:
        for i in range(1, len(predicted_hist)):
            prev, curr = predicted_hist[i-1], predicted_hist[i]
            pt1 = (int(prev["x"]), int(prev["y"]))
            pt2 = (int(curr["x"]), int(curr["y"]))
            cv2.line(debug, pt1, pt2, (200, 100, 0), 1, cv2.LINE_AA)
    
    # Stats overlay
    cv2.rectangle(debug, (5, 5), (450, 120), (0, 0, 0), -1)
    cv2.putText(debug, "DEBUG MODE", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(debug, f"Persons: {len(filter_result.get('all_persons', []))}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(debug, f"Rejected: {len(filter_result.get('rejected_persons', []))}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(debug, f"Ball candidates: {len(ball_result.get('all_candidates', []))}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(debug, f"Ball status: {ball_result.get('status', 'NONE')}", (10, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return debug


def process_video(args):
    """Main processing pipeline."""
    print("="*70)
    print("TrackScore Core Perception - tennis_match3.mp4")
    print("="*70)
    
    # Load calibration
    print("\n[1/5] Loading court calibration...")
    court = CourtCalibration(args.calibration)
    
    # Load video
    print("\n[2/5] Loading video...")
    loader = VideoLoader(args.video)
    metadata = loader.get_metadata()
    fps, width, height = metadata["fps"], metadata["width"], metadata["height"]
    total_frames = metadata["frame_count"]
    print(f"  Resolution: {width}x{height} @ {fps}fps")
    print(f"  Duration: {metadata['duration_seconds']:.1f}s ({total_frames} frames)")
    
    max_frames = int(args.max_seconds * fps) if args.max_seconds else total_frames
    
    # Initialize detectors
    print("\n[3/5] Initializing detectors...")
    player_detector = PlayerDetector()
    player_tracker = PlayerTracker()
    player_filter = RobustPlayerFilter()
    ball_detector = TennisBallDetector(court)
    
    # Prepare output videos
    print("\n[4/5] Processing frames...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Main output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Debug output
    debug_path = output_path.parent / output_path.name.replace(".mp4", "_debug.mp4")
    debug_out = cv2.VideoWriter(str(debug_path), fourcc, fps, (width, height))
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    
    # Statistics
    stats = {
        "player_a_frames": 0,
        "player_b_frames": 0,
        "both_players_frames": 0,
        "ball_confirmed": 0,
        "ball_predicted": 0,
        "ball_lost": 0,
        "persons_rejected": 0,
        "far_court_candidates_total": 0,
        "near_court_candidates_total": 0,
        "trajectory_segments": 0,
        "implausible_jumps_rejected": 0,
        "ball_speed_samples_accepted": 0,
        "ball_speed_samples_rejected": 0,
    }
    
    player_a_history = []
    player_b_history = []
    
    frame_number = 0
    processed = 0
    start_time = time.time()
    prev_frame_gray = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or processed >= max_frames:
                break
            
            timestamp = frame_number / fps
            
            # Detect and track persons
            person_result = player_detector.detect_players(frame)
            persons = person_result.get("players", [])
            
            tracking_result = player_tracker.process_frame(frame, frame_number, timestamp)
            tracked_persons = tracking_result.get("players", [])
            
            # Filter to actual tennis players
            filter_result = player_filter.filter_persons(tracked_persons, height, width)
            
            # Detect ball (with prev frame for motion detection)
            ball_result = ball_detector.detect_ball(frame, frame_number, timestamp, prev_frame_gray)
            
            # Update player history
            player_a = filter_result.get("player_a")
            player_b = filter_result.get("player_b")
            
            if player_a:
                player_a_history.append({
                    "timestamp": timestamp,
                    "foot_pos": player_a["foot_position"]
                })
            if player_b:
                player_b_history.append({
                    "timestamp": timestamp,
                    "foot_pos": player_b["foot_position"]
                })
            
            # Calculate speeds
            player_a_speed = calculate_player_speed(court, player_a_history) if player_a else None
            player_b_speed = calculate_player_speed(court, player_b_history) if player_b else None
            ball_speed = calculate_ball_speed(court, ball_result.get("confirmed_trajectory", []))
            
            # Create output frames
            output_frame = court.draw_court_overlay(frame)
            
            # Draw players
            if player_a:
                bbox = player_a.get("bbox", {})
                if isinstance(bbox, dict):
                    x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                    x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = "Player A"
                    if player_a_speed:
                        label += f" | Est. {player_a_speed:.1f} km/h"
                    cv2.putText(output_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if player_b:
                bbox = player_b.get("bbox", {})
                if isinstance(bbox, dict):
                    x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                    x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = "Player B"
                    if player_b_speed:
                        label += f" | Est. {player_b_speed:.1f} km/h"
                    cv2.putText(output_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw ball
            ball = ball_result.get("ball")
            if ball:
                bx, by = int(ball["x"]), int(ball["y"])
                is_pred = ball.get("predicted", False)
                
                if is_pred:
                    # Blue for predictions
                    color = (255, 0, 0)
                else:
                    # Green for confirmed
                    color = (0, 255, 0)
                
                cv2.circle(output_frame, (bx, by), 8, color, 2)
                
                # Only show speed for confirmed detections
                if ball_speed and not is_pred:
                    cv2.putText(output_frame, f"Est. {ball_speed:.0f} km/h", (bx+15, by-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw CONFIRMED trajectory only (no long prediction zig-zags)
            confirmed_traj = ball_result.get("confirmed_trajectory", [])
            if len(confirmed_traj) >= 2:
                for i in range(1, len(confirmed_traj)):
                    prev, curr = confirmed_traj[i-1], confirmed_traj[i]
                    
                    # Only draw within same segment
                    if prev.get("segment_id") == curr.get("segment_id"):
                        pt1 = (int(prev["x"]), int(prev["y"]))
                        pt2 = (int(curr["x"]), int(curr["y"]))
                        cv2.line(output_frame, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)
            
            # Write outputs
            out.write(output_frame)
            
            # Debug frame
            debug_frame = draw_debug_frame(frame, court, filter_result, ball_result)
            debug_out.write(debug_frame)
            
            # Update stats
            stats["persons_rejected"] += len(filter_result.get("rejected_persons", []))
            stats["far_court_candidates_total"] += len(filter_result.get("far_court_candidates", []))
            stats["near_court_candidates_total"] += len(filter_result.get("near_court_candidates", []))
            
            if player_a:
                stats["player_a_frames"] += 1
            if player_b:
                stats["player_b_frames"] += 1
            if player_a and player_b:
                stats["both_players_frames"] += 1
            
            # Ball statistics
            ball_status = ball_result.get("status", "LOST")
            if ball and not ball.get("predicted"):
                stats["ball_confirmed"] += 1
            elif ball and ball.get("predicted"):
                stats["ball_predicted"] += 1
            else:
                stats["ball_lost"] += 1
            
            # Track speed calculation success
            if ball_speed:
                stats["ball_speed_samples_accepted"] += 1
            
            # Update prev_frame_gray for next iteration
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame_number += 1
            processed += 1
            
            if processed % 30 == 0:
                print(f"  Processed {processed}/{max_frames} frames...")
    
    finally:
        cap.release()
        out.release()
        debug_out.release()
    
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\n[5/5] Processing complete: {processed} frames in {elapsed:.1f}s")
    print(f"\n{'='*70}")
    print("CORE PERCEPTION STATISTICS")
    print(f"{'='*70}")
    print(f"\nPLAYER DETECTION:")
    print(f"  Player A visible: {stats['player_a_frames']}/{processed} ({100*stats['player_a_frames']/processed:.1f}%)")
    print(f"    Far court candidates: {stats['far_court_candidates_total']} total")
    print(f"  Player B visible: {stats['player_b_frames']}/{processed} ({100*stats['player_b_frames']/processed:.1f}%)")
    print(f"    Near court candidates: {stats['near_court_candidates_total']} total")
    print(f"  Both players: {stats['both_players_frames']}/{processed} ({100*stats['both_players_frames']/processed:.1f}%)")
    print(f"  Persons rejected: {stats['persons_rejected']}")
    
    print(f"\nBALL DETECTION (NEW V4 PIPELINE):")
    print(f"  CONFIRMED direct: {stats['ball_confirmed']}/{processed} ({100*stats['ball_confirmed']/processed:.1f}%)")
    print(f"  PREDICTED (≤3 frames): {stats['ball_predicted']}/{processed} ({100*stats['ball_predicted']/processed:.1f}%)")
    print(f"  LOST: {stats['ball_lost']}/{processed} ({100*stats['ball_lost']/processed:.1f}%)")
    print(f"  Total visible: {stats['ball_confirmed']+stats['ball_predicted']}/{processed} ({100*(stats['ball_confirmed']+stats['ball_predicted'])/processed:.1f}%)")
    
    # Get trajectory segment count
    if ball_detector.confirmed_trajectory:
        segments = set(t.get("segment_id", 0) for t in ball_detector.confirmed_trajectory)
        print(f"  Trajectory segments: {len(segments)}")
        print(f"  Confirmed trajectory points: {len(ball_detector.confirmed_trajectory)}")
    
    print(f"\nBALL SPEED CALCULATION:")
    print(f"  Frames with valid speed: {stats['ball_speed_samples_accepted']}")
    
    print(f"\nOutput video: {output_path}")
    print(f"Debug video: {debug_path}")
    print(f"{'='*70}")
    print("\n⚠️  VISUAL INSPECTION REQUIRED:")
    print("  1. Check that ball marker follows real ball (not court lines/players)")
    print("  2. Verify no long prediction zig-zags")
    print("  3. Confirm trajectory stays on actual ball path")
    print("  4. Check ball speed estimates are plausible when shown")
    print(f"{'='*70}")
    
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--calibration", required=True, help="Court calibration JSON")
    parser.add_argument("--output", default="outputs/final/tennis_match3_core.mp4")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    
    return process_video(args)


if __name__ == "__main__":
    sys.exit(main())
