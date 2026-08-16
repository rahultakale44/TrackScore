"""
Robust player filtering for singles tennis using court geometry.

Filters YOLO person detections to identify exactly 2 tennis competitors.
Rejects ball boys, officials, spectators, and umpire.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class PlayerFilterConfig:
    """Configuration for robust player filtering."""
    
    # Court regions (fractions of frame height)
    far_court_y_min: float = 0.10  # Top 10% likely has spectators
    far_court_y_max: float = 0.50  # Player A zone - EXPANDED from 0.45
    near_court_y_min: float = 0.50  # Player B zone - ADJUSTED to overlap
    near_court_y_max: float = 0.90  # Bottom 10% may have ball boys
    
    # Minimum person size (reject distant spectators)
    min_bbox_height_ratio: float = 0.06  # REDUCED from 0.08 for distant far-court player
    min_bbox_width_ratio: float = 0.02  # REDUCED from 0.03 for distant far-court player
    
    # Temporal persistence - INCREASED for better identity retention
    max_missing_frames: int = 30  # Retain player for 30 frames if lost (was 15)
    
    # Position jump threshold (pixels)
    max_position_jump: float = 200.0  # Max movement between frames


class RobustPlayerFilter:
    """
    Filter person detections to find exactly 2 tennis players.
    
    Strategy:
    1. Reject persons outside playable court regions
    2. Reject persons with wrong size/aspect
    3. Select best candidate from far-court zone → Player A
    4. Select best candidate from near-court zone → Player B
    5. Maintain identity across frames with tracking
    """
    
    def __init__(self, config: Optional[PlayerFilterConfig] = None):
        self.config = config if config is not None else PlayerFilterConfig()
        
        # Player state tracking
        self.player_a_last_pos: Optional[Tuple[float, float]] = None
        self.player_b_last_pos: Optional[Tuple[float, float]] = None
        self.player_a_missing_count: int = 0
        self.player_b_missing_count: int = 0
        
        # Track history for velocity estimation
        self.player_a_history: List[Dict[str, Any]] = []
        self.player_b_history: List[Dict[str, Any]] = []
    
    def _get_foot_position(self, person: Dict[str, Any]) -> Tuple[float, float]:
        """Extract foot position from person detection."""
        bbox = person.get("bbox", {})
        
        if isinstance(bbox, dict):
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        elif isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            # Fallback
            foot = person.get("foot_point", {})
            return (float(foot.get("x", 0)), float(foot.get("y", 0)))
        
        # Foot at bottom-center of bbox
        return ((x1 + x2) / 2.0, float(y2))
    
    def _get_bbox_size(self, person: Dict[str, Any]) -> Tuple[float, float]:
        """Get bbox width and height."""
        bbox = person.get("bbox", {})
        
        if isinstance(bbox, dict):
            return (float(bbox.get("width", 0)), float(bbox.get("height", 0)))
        elif isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            return (abs(x2 - x1), abs(y2 - y1))
        
        return (0.0, 0.0)
    
    def filter_persons(
        self,
        person_detections: List[Dict[str, Any]],
        frame_height: int,
        frame_width: int,
    ) -> Dict[str, Any]:
        """
        Filter person detections to find Player A and Player B.
        
        Returns:
            {
                "all_persons": [...],
                "rejected_persons": [...],  # with rejection reasons
                "far_court_candidates": [...],
                "near_court_candidates": [...],
                "player_a": {...} or None,
                "player_b": {...} or None,
            }
        """
        all_persons = list(person_detections)
        rejected = []
        far_court_candidates = []
        near_court_candidates = []
        
        # Calculate region boundaries
        far_y_min = frame_height * self.config.far_court_y_min
        far_y_max = frame_height * self.config.far_court_y_max
        near_y_min = frame_height * self.config.near_court_y_min
        near_y_max = frame_height * self.config.near_court_y_max
        
        min_height = frame_height * self.config.min_bbox_height_ratio
        min_width = frame_width * self.config.min_bbox_width_ratio
        
        # Filter each person
        for person in person_detections:
            foot_x, foot_y = self._get_foot_position(person)
            bbox_w, bbox_h = self._get_bbox_size(person)
            
            person_data = dict(person)
            person_data["foot_position"] = (foot_x, foot_y)
            person_data["bbox_size"] = (bbox_w, bbox_h)
            
            # Check size filters
            if bbox_h < min_height:
                person_data["rejection_reason"] = "too_small"
                rejected.append(person_data)
                continue
            
            if bbox_w < min_width:
                person_data["rejection_reason"] = "too_narrow"
                rejected.append(person_data)
                continue
            
            # Check vertical position
            in_far_zone = far_y_min <= foot_y <= far_y_max
            in_near_zone = near_y_min <= foot_y <= near_y_max
            
            if in_far_zone:
                far_court_candidates.append(person_data)
            elif in_near_zone:
                near_court_candidates.append(person_data)
            else:
                if foot_y < far_y_min:
                    person_data["rejection_reason"] = "above_far_court"
                elif foot_y > near_y_max:
                    person_data["rejection_reason"] = "below_near_court"
                else:
                    person_data["rejection_reason"] = "middle_zone_excluded"
                rejected.append(person_data)
        
        # Select Player A from far court
        player_a = self._select_player(
            far_court_candidates,
            self.player_a_last_pos,
            "Player A"
        )
        
        # If no candidate but we have recent history, preserve identity temporarily
        if not player_a and self.player_a_last_pos and self.player_a_missing_count < self.config.max_missing_frames:
            # Keep tracking even without detection
            pass  # Will update missing count below
        
        # Select Player B from near court
        player_b = self._select_player(
            near_court_candidates,
            self.player_b_last_pos,
            "Player B"
        )
        
        # If no candidate but we have recent history, preserve identity temporarily  
        if not player_b and self.player_b_last_pos and self.player_b_missing_count < self.config.max_missing_frames:
            # Keep tracking even without detection
            pass  # Will update missing count below
        
        # Update player state
        if player_a:
            self.player_a_last_pos = player_a["foot_position"]
            self.player_a_missing_count = 0
            self.player_a_history.append({
                "position": player_a["foot_position"],
                "bbox": player_a.get("bbox"),
            })
            if len(self.player_a_history) > 30:
                self.player_a_history = self.player_a_history[-30:]
        else:
            self.player_a_missing_count += 1
            if self.player_a_missing_count > self.config.max_missing_frames:
                self.player_a_last_pos = None
        
        if player_b:
            self.player_b_last_pos = player_b["foot_position"]
            self.player_b_missing_count = 0
            self.player_b_history.append({
                "position": player_b["foot_position"],
                "bbox": player_b.get("bbox"),
            })
            if len(self.player_b_history) > 30:
                self.player_b_history = self.player_b_history[-30:]
        else:
            self.player_b_missing_count += 1
            if self.player_b_missing_count > self.config.max_missing_frames:
                self.player_b_last_pos = None
        
        return {
            "all_persons": all_persons,
            "rejected_persons": rejected,
            "far_court_candidates": far_court_candidates,
            "near_court_candidates": near_court_candidates,
            "player_a": player_a,
            "player_b": player_b,
        }
    
    def _select_player(
        self,
        candidates: List[Dict[str, Any]],
        last_position: Optional[Tuple[float, float]],
        label: str
    ) -> Optional[Dict[str, Any]]:
        """
        Select best player candidate from list.
        
        Prefers:
        1. Closest to last known position (temporal continuity)
        2. Highest confidence
        3. Largest bbox (more prominent in frame)
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            selected = dict(candidates[0])
            selected["player_label"] = label
            selected["selection_reason"] = "only_candidate"
            return selected
        
        # Score each candidate
        scored = []
        for candidate in candidates:
            conf = candidate.get("confidence", 0.5)
            bbox_w, bbox_h = candidate["bbox_size"]
            bbox_area = bbox_w * bbox_h
            foot_pos = candidate["foot_position"]
            
            # Base score from confidence and size
            score = conf * 0.3 + np.sqrt(bbox_area) * 0.001
            
            # Temporal continuity bonus
            if last_position is not None:
                dist = np.hypot(
                    foot_pos[0] - last_position[0],
                    foot_pos[1] - last_position[1]
                )
                
                if dist < self.config.max_position_jump:
                    # Strong bonus for nearby candidates
                    proximity_bonus = (1.0 - dist / self.config.max_position_jump) * 0.7
                    score += proximity_bonus
            
            scored.append((score, candidate))
        
        # Select highest score
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = dict(scored[0][1])
        selected["player_label"] = label
        selected["selection_score"] = scored[0][0]
        selected["selection_reason"] = "best_scored"
        
        return selected
    
    def reset(self):
        """Reset player tracking state."""
        self.player_a_last_pos = None
        self.player_b_last_pos = None
        self.player_a_missing_count = 0
        self.player_b_missing_count = 0
        self.player_a_history.clear()
        self.player_b_history.clear()
