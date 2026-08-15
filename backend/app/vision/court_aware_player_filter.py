"""
Court-aware player filtering for tennis match analysis.

Uses calibrated court coordinates to distinguish actual tennis players
from ball boys, officials, spectators, and other non-players.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .court_calibrator import CourtCalibrator


class PlayerFilterError(Exception):
    """Raised when player filtering fails."""


@dataclass
class PlayerFilterConfig:
    """Configuration for court-aware player filtering."""
    
    # Court membership
    court_margin_meters: float = 15.0  # Large margin for now - homography not perfect
    
    # Behind-baseline rejection (ball boys, etc.)
    max_behind_baseline_meters: float = 10.0  # Increased tolerance
    
    # Minimum time a player must be visible to be considered
    minimum_appearance_frames: int = 10
    
    # Temporal consistency
    position_jump_threshold_meters: float = 8.0  # Max movement per frame at 30fps
    
    # Player count
    expected_players: int = 2  # Singles match


class CourtAwarePlayerFilter:
    """
    Filter person detections to identify actual tennis players.
    
    Uses court calibration to:
    - Reject people outside court boundaries
    - Reject ball boys (far behind baseline)
    - Reject line judges (on sidelines)
    - Reject spectators (beyond court margin)
    - Select 2 players on opposite sides of net
    """
    
    def __init__(
        self,
        court_calibrator: CourtCalibrator,
        config: PlayerFilterConfig | None = None
    ):
        if court_calibrator.homography_matrix is None:
            raise PlayerFilterError("Court calibrator must have valid homography")
        
        self.calibrator = court_calibrator
        self.config = config if config is not None else PlayerFilterConfig()
        
        # Track player identities across frames
        self.player_a_history: List[Dict[str, Any]] = []
        self.player_b_history: List[Dict[str, Any]] = []
        
        # Track which ByteTrack IDs correspond to Player A/B
        self.player_a_track_ids: set = set()
        self.player_b_track_ids: set = set()
    
    def _get_foot_position(self, track: Dict[str, Any]) -> Tuple[float, float]:
        """Extract foot position from track."""
        foot_point = track.get("foot_point")
        if foot_point:
            return (float(foot_point["x"]), float(foot_point["y"]))
        
        # Fallback to bbox bottom center
        bbox = track.get("bbox", {})
        if isinstance(bbox, dict):
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
            return ((x1 + x2) / 2.0, y2)
        
        return (0.0, 0.0)
    
    def filter_person_detections(
        self,
        person_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Filter raw person detections to find tennis players.
        
        Returns:
            {
                "all_persons": [...],  # All detected persons with metadata
                "court_persons": [...],  # Persons inside court region
                "rejected_persons": [...],  # Persons rejected with reasons
                "player_candidates": [...],  # Final player candidates
            }
        """
        all_persons = []
        court_persons = []
        rejected_persons = []
        
        for track in person_tracks:
            foot_image = self._get_foot_position(track)
            court_coords = self.calibrator.image_to_court_coordinates(foot_image)
            
            # Add court coordinates to track
            person = dict(track)
            person["foot_image"] = foot_image
            person["foot_court"] = court_coords
            
            all_persons.append(person)
            
            # Check if transformation succeeded
            if court_coords is None:
                person["rejection_reason"] = "homography_failed"
                rejected_persons.append(person)
                continue
            
            x_court, y_court = court_coords
            
            # DEBUG: Print court coordinates for first few frames
            import os
            if os.environ.get("DEBUG_COURT_COORDS") == "1":
                print(f"      Person at image ({foot_image[0]:.0f}, {foot_image[1]:.0f}) → court ({x_court:.2f}, {y_court:.2f})")
            
            # Check court membership with larger margin for now
            if not self.calibrator.is_point_in_court(
                foot_image,
                margin_meters=self.config.court_margin_meters
            ):
                person["rejection_reason"] = "outside_court_bounds"
                person["debug_court_coords"] = (x_court, y_court)
                rejected_persons.append(person)
                continue
            
            # Check if too far behind baseline (ball boy detection)
            court_length = self.calibrator.court_model.COURT_LENGTH
            if y_court < -self.config.max_behind_baseline_meters:
                person["rejection_reason"] = "behind_near_baseline"
                rejected_persons.append(person)
                continue
            
            if y_court > court_length + self.config.max_behind_baseline_meters:
                person["rejection_reason"] = "behind_far_baseline"
                rejected_persons.append(person)
                continue
            
            # Passed all filters - this is a valid court person
            court_persons.append(person)
        
        # Select final players
        player_candidates = self._select_players(court_persons)
        
        return {
            "all_persons": all_persons,
            "court_persons": court_persons,
            "rejected_persons": rejected_persons,
            "player_candidates": player_candidates,
        }
    
    def _select_players(
        self,
        court_persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select 2 players from court persons.
        
        Strategy:
        1. Prefer persons on opposite sides of net
        2. Use temporal consistency (match previous Player A/B)
        3. Select largest/most confident if ambiguous
        """
        if len(court_persons) == 0:
            return []
        
        if len(court_persons) == 1:
            # Only one person - assign to Player A or B based on history
            return self._assign_single_player(court_persons[0])
        
        if len(court_persons) == 2:
            # Two persons - assign based on net side
            return self._assign_two_players(court_persons)
        
        # More than 2 persons - need to disambiguate
        return self._select_best_two_players(court_persons)
    
    def _assign_single_player(
        self,
        person: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assign single visible player to Player A or B."""
        court_side = self.calibrator.get_court_side(person["foot_image"])
        track_id = person.get("track_id")
        
        # Check if this track_id is known
        if track_id in self.player_a_track_ids:
            person["player_label"] = "Player A"
            person["court_side"] = court_side
            self.player_a_history.append(person)
            return [person]
        
        if track_id in self.player_b_track_ids:
            person["player_label"] = "Player B"
            person["court_side"] = court_side
            self.player_b_history.append(person)
            return [person]
        
        # Unknown track - assign based on court side and history
        if court_side == "far":
            person["player_label"] = "Player A"
            self.player_a_track_ids.add(track_id)
            self.player_a_history.append(person)
        else:
            person["player_label"] = "Player B"
            self.player_b_track_ids.add(track_id)
            self.player_b_history.append(person)
        
        person["court_side"] = court_side
        return [person]
    
    def _assign_two_players(
        self,
        persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assign two persons to Player A and B."""
        # Sort by court Y coordinate (far to near)
        sorted_persons = sorted(
            persons,
            key=lambda p: p["foot_court"][1] if p["foot_court"] else 0,
            reverse=True  # Higher Y = farther from camera
        )
        
        far_person = sorted_persons[0]
        near_person = sorted_persons[1]
        
        # Assign labels
        far_person["player_label"] = "Player A"
        far_person["court_side"] = "far"
        self.player_a_track_ids.add(far_person.get("track_id"))
        self.player_a_history.append(far_person)
        
        near_person["player_label"] = "Player B"
        near_person["court_side"] = "near"
        self.player_b_track_ids.add(near_person.get("track_id"))
        self.player_b_history.append(near_person)
        
        return [far_person, near_person]
    
    def _select_best_two_players(
        self,
        court_persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select best 2 players when more than 2 candidates exist.
        
        Uses:
        - Temporal consistency (known track IDs)
        - Confidence scores
        - Bounding box size
        - Court position
        """
        # Score each candidate
        scored = []
        for person in court_persons:
            track_id = person.get("track_id")
            confidence = person.get("confidence", 0.0)
            bbox = person.get("bbox", {})
            
            if isinstance(bbox, dict):
                area = bbox.get("width", 0) * bbox.get("height", 0)
            else:
                area = 0
            
            # Temporal consistency bonus
            temporal_bonus = 0.0
            if track_id in self.player_a_track_ids or track_id in self.player_b_track_ids:
                temporal_bonus = 0.5
            
            # Combined score
            score = confidence * 0.4 + np.sqrt(area) * 0.001 + temporal_bonus
            
            scored.append((score, person))
        
        # Select top 2
        scored.sort(key=lambda x: x[0], reverse=True)
        top_two = [person for score, person in scored[:2]]
        
        # Assign labels
        return self._assign_two_players(top_two)
    
    def reset(self) -> None:
        """Reset player tracking state."""
        self.player_a_history.clear()
        self.player_b_history.clear()
        self.player_a_track_ids.clear()
        self.player_b_track_ids.clear()
