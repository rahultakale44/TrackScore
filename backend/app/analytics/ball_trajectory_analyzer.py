from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class BallTrajectoryAnalysisError(Exception):
    """Raised when ball trajectory analysis fails."""


@dataclass
class BallTrajectoryConfig:
    """
    Configuration for tennis-ball trajectory analysis.
    """

    minimum_points_for_analysis: int = 3

    minimum_time_delta: float = 0.001

    minimum_motion_pixels: float = 2.0

    minimum_direction_change_degrees: float = 20.0

    strong_direction_change_degrees: float = 45.0

    maximum_prediction_ratio: float = 0.50

    bounce_vertical_reversal_bonus: float = 0.45

    direction_change_weight: float = 0.35

    detection_quality_weight: float = 0.20

    bounce_score_threshold: float = 0.55

    history_limit: int = 300


class BallTrajectoryAnalyzer:
    """
    Analyses tracked tennis-ball positions over time.

    Features:
    - displacement
    - pixel velocity
    - trajectory direction
    - turning angle
    - vertical direction reversal
    - bounce candidate detection
    - bounce confidence score

    Image coordinate convention:

        x increases → right
        y increases → downward

    Therefore a typical bounce can sometimes appear as:

        downward motion: velocity_y > 0
                ↓
             BOUNCE
                ↓
        upward motion: velocity_y < 0

    This is only one signal and not treated as absolute truth.
    """

    def __init__(
        self,
        config: BallTrajectoryConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else BallTrajectoryConfig()
        )

        self._validate_config()

        self.motion_history: List[
            Dict[str, Any]
        ] = []

        self.bounce_candidates: List[
            Dict[str, Any]
        ] = []

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if config.minimum_points_for_analysis < 3:
            raise BallTrajectoryAnalysisError(
                "minimum_points_for_analysis must be at least 3."
            )

        if config.minimum_time_delta <= 0:
            raise BallTrajectoryAnalysisError(
                "minimum_time_delta must be greater than zero."
            )

        if config.minimum_motion_pixels < 0:
            raise BallTrajectoryAnalysisError(
                "minimum_motion_pixels cannot be negative."
            )

        if not (
            0.0
            <= config.minimum_direction_change_degrees
            <= 180.0
        ):
            raise BallTrajectoryAnalysisError(
                "minimum_direction_change_degrees must be within [0, 180]."
            )

        if not (
            0.0
            <= config.strong_direction_change_degrees
            <= 180.0
        ):
            raise BallTrajectoryAnalysisError(
                "strong_direction_change_degrees must be within [0, 180]."
            )

        if not (
            0.0
            <= config.maximum_prediction_ratio
            <= 1.0
        ):
            raise BallTrajectoryAnalysisError(
                "maximum_prediction_ratio must be within [0, 1]."
            )

        if not (
            0.0
            <= config.bounce_score_threshold
            <= 1.0
        ):
            raise BallTrajectoryAnalysisError(
                "bounce_score_threshold must be within [0, 1]."
            )

        if config.history_limit <= 0:
            raise BallTrajectoryAnalysisError(
                "history_limit must be greater than zero."
            )

    # ============================================================
    # POINT VALIDATION
    # ============================================================

    @staticmethod
    def validate_point(
        point: Dict[str, Any],
    ) -> None:
        required_fields = [
            "frame_number",
            "timestamp_seconds",
            "x",
            "y",
        ]

        for field in required_fields:
            if field not in point:
                raise BallTrajectoryAnalysisError(
                    f"Trajectory point missing '{field}'."
                )

    # ============================================================
    # BASIC GEOMETRY
    # ============================================================

    @staticmethod
    def calculate_distance(
        point_a: Sequence[float],
        point_b: Sequence[float],
    ) -> float:
        if len(point_a) != 2:
            raise BallTrajectoryAnalysisError(
                "point_a must contain x and y."
            )

        if len(point_b) != 2:
            raise BallTrajectoryAnalysisError(
                "point_b must contain x and y."
            )

        dx = (
            float(point_b[0])
            - float(point_a[0])
        )

        dy = (
            float(point_b[1])
            - float(point_a[1])
        )

        return float(
            hypot(
                dx,
                dy,
            )
        )

    @staticmethod
    def calculate_velocity(
        point_a: Dict[str, Any],
        point_b: Dict[str, Any],
        minimum_time_delta: float = 0.001,
    ) -> Tuple[float, float]:
        """
        Calculate velocity in pixels/second.
        """

        BallTrajectoryAnalyzer.validate_point(
            point_a
        )

        BallTrajectoryAnalyzer.validate_point(
            point_b
        )

        delta_time = (
            float(
                point_b["timestamp_seconds"]
            )
            - float(
                point_a["timestamp_seconds"]
            )
        )

        if delta_time < minimum_time_delta:
            return (
                0.0,
                0.0,
            )

        velocity_x = (
            float(point_b["x"])
            - float(point_a["x"])
        ) / delta_time

        velocity_y = (
            float(point_b["y"])
            - float(point_a["y"])
        ) / delta_time

        return (
            float(velocity_x),
            float(velocity_y),
        )

    @staticmethod
    def calculate_speed(
        velocity: Sequence[float],
    ) -> float:
        if len(velocity) != 2:
            raise BallTrajectoryAnalysisError(
                "Velocity must contain x and y."
            )

        return float(
            hypot(
                float(velocity[0]),
                float(velocity[1]),
            )
        )

    # ============================================================
    # VECTOR ANGLE
    # ============================================================

    @staticmethod
    def calculate_direction_change(
        velocity_before: Sequence[float],
        velocity_after: Sequence[float],
    ) -> float:
        """
        Calculate angle between two velocity vectors.
        """

        if (
            len(velocity_before) != 2
            or len(velocity_after) != 2
        ):
            raise BallTrajectoryAnalysisError(
                "Velocity vectors must contain x and y."
            )

        vector_a = np.array(
            velocity_before,
            dtype=np.float64,
        )

        vector_b = np.array(
            velocity_after,
            dtype=np.float64,
        )

        magnitude_a = float(
            np.linalg.norm(
                vector_a
            )
        )

        magnitude_b = float(
            np.linalg.norm(
                vector_b
            )
        )

        if (
            magnitude_a == 0.0
            or magnitude_b == 0.0
        ):
            return 0.0

        cosine = float(
            np.dot(
                vector_a,
                vector_b,
            )
            / (
                magnitude_a
                * magnitude_b
            )
        )

        cosine = float(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )

        angle = degrees(
            acos(
                cosine
            )
        )

        return round(
            float(angle),
            3,
        )

    # ============================================================
    # VERTICAL REVERSAL
    # ============================================================

    @staticmethod
    def has_vertical_reversal(
        velocity_before: Sequence[float],
        velocity_after: Sequence[float],
    ) -> bool:
        """
        Detect downward -> upward reversal.

        Image Y:
            positive = downward
            negative = upward
        """

        before_y = float(
            velocity_before[1]
        )

        after_y = float(
            velocity_after[1]
        )

        return (
            before_y > 0.0
            and after_y < 0.0
        )

    # ============================================================
    # PREDICTION QUALITY
    # ============================================================

    @staticmethod
    def calculate_prediction_ratio(
        points: List[
            Dict[str, Any]
        ],
    ) -> float:
        if not points:
            return 0.0

        predicted = sum(
            1
            for point in points
            if bool(
                point.get(
                    "predicted",
                    False,
                )
            )
        )

        return float(
            predicted
            / len(points)
        )

    # ============================================================
    # BOUNCE SCORE
    # ============================================================

    def calculate_bounce_score(
        self,
        previous_point: Dict[str, Any],
        current_point: Dict[str, Any],
        next_point: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyse a three-point trajectory window.
        """

        velocity_before = (
            self.calculate_velocity(
                previous_point,
                current_point,
                self.config.minimum_time_delta,
            )
        )

        velocity_after = (
            self.calculate_velocity(
                current_point,
                next_point,
                self.config.minimum_time_delta,
            )
        )

        speed_before = (
            self.calculate_speed(
                velocity_before
            )
        )

        speed_after = (
            self.calculate_speed(
                velocity_after
            )
        )

        direction_change = (
            self.calculate_direction_change(
                velocity_before,
                velocity_after,
            )
        )

        vertical_reversal = (
            self.has_vertical_reversal(
                velocity_before,
                velocity_after,
            )
        )

        local_points = [
            previous_point,
            current_point,
            next_point,
        ]

        prediction_ratio = (
            self.calculate_prediction_ratio(
                local_points
            )
        )

        score = 0.0

        if vertical_reversal:
            score += (
                self.config
                .bounce_vertical_reversal_bonus
            )

        normalized_direction_change = min(
            direction_change
            / max(
                self.config
                .strong_direction_change_degrees,
                1.0,
            ),
            1.0,
        )

        score += (
            normalized_direction_change
            * self.config
            .direction_change_weight
        )

        detection_quality = (
            1.0
            - prediction_ratio
        )

        score += (
            detection_quality
            * self.config
            .detection_quality_weight
        )

        score = min(
            max(
                score,
                0.0,
            ),
            1.0,
        )

        motion_before = (
            self.calculate_distance(
                (
                    previous_point["x"],
                    previous_point["y"],
                ),
                (
                    current_point["x"],
                    current_point["y"],
                ),
            )
        )

        motion_after = (
            self.calculate_distance(
                (
                    current_point["x"],
                    current_point["y"],
                ),
                (
                    next_point["x"],
                    next_point["y"],
                ),
            )
        )

        sufficient_motion = (
            motion_before
            >= self.config.minimum_motion_pixels
            and
            motion_after
            >= self.config.minimum_motion_pixels
        )

        direction_changed = (
            direction_change
            >= self.config
            .minimum_direction_change_degrees
        )

        reliable_window = (
            prediction_ratio
            <= self.config
            .maximum_prediction_ratio
        )

        is_bounce_candidate = (
            sufficient_motion
            and direction_changed
            and reliable_window
            and score
            >= self.config
            .bounce_score_threshold
        )

        return {
            "frame_number": int(
                current_point[
                    "frame_number"
                ]
            ),

            "timestamp_seconds": round(
                float(
                    current_point[
                        "timestamp_seconds"
                    ]
                ),
                3,
            ),

            "position": {
                "x": round(
                    float(
                        current_point["x"]
                    ),
                    3,
                ),
                "y": round(
                    float(
                        current_point["y"]
                    ),
                    3,
                ),
            },

            "velocity_before": {
                "x": round(
                    velocity_before[0],
                    3,
                ),
                "y": round(
                    velocity_before[1],
                    3,
                ),
            },

            "velocity_after": {
                "x": round(
                    velocity_after[0],
                    3,
                ),
                "y": round(
                    velocity_after[1],
                    3,
                ),
            },

            "speed_before_pixels_per_second": round(
                speed_before,
                3,
            ),

            "speed_after_pixels_per_second": round(
                speed_after,
                3,
            ),

            "direction_change_degrees": round(
                direction_change,
                3,
            ),

            "vertical_reversal": (
                vertical_reversal
            ),

            "prediction_ratio": round(
                prediction_ratio,
                3,
            ),

            "sufficient_motion": (
                sufficient_motion
            ),

            "reliable_window": (
                reliable_window
            ),

            "bounce_score": round(
                score,
                4,
            ),

            "is_bounce_candidate": (
                is_bounce_candidate
            ),
        }

    # ============================================================
    # FULL TRAJECTORY ANALYSIS
    # ============================================================

    def analyse_trajectory(
        self,
        trajectory: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Analyse complete trajectory and detect bounce candidates.
        """

        if trajectory is None:
            raise BallTrajectoryAnalysisError(
                "Trajectory cannot be None."
            )

        if (
            len(trajectory)
            < self.config.minimum_points_for_analysis
        ):
            return {
                "trajectory_points": len(
                    trajectory
                ),
                "analysed_windows": 0,
                "bounce_candidate_count": 0,
                "bounce_candidates": [],
                "motion_windows": [],
            }

        motion_windows = []

        bounce_candidates = []

        for index in range(
            1,
            len(trajectory) - 1,
        ):
            previous_point = (
                trajectory[
                    index - 1
                ]
            )

            current_point = (
                trajectory[
                    index
                ]
            )

            next_point = (
                trajectory[
                    index + 1
                ]
            )

            analysis = (
                self.calculate_bounce_score(
                    previous_point,
                    current_point,
                    next_point,
                )
            )

            motion_windows.append(
                analysis
            )

            if analysis[
                "is_bounce_candidate"
            ]:
                bounce_candidates.append(
                    analysis
                )

        self.motion_history.extend(
            motion_windows
        )

        self.bounce_candidates.extend(
            bounce_candidates
        )

        if (
            len(self.motion_history)
            > self.config.history_limit
        ):
            self.motion_history = (
                self.motion_history[
                    -self.config.history_limit:
                ]
            )

        if (
            len(self.bounce_candidates)
            > self.config.history_limit
        ):
            self.bounce_candidates = (
                self.bounce_candidates[
                    -self.config.history_limit:
                ]
            )

        return {
            "trajectory_points": len(
                trajectory
            ),

            "analysed_windows": len(
                motion_windows
            ),

            "bounce_candidate_count": len(
                bounce_candidates
            ),

            "bounce_candidates": (
                bounce_candidates
            ),

            "motion_windows": (
                motion_windows
            ),
        }

    # ============================================================
    # DE-DUPLICATION
    # ============================================================

    @staticmethod
    def deduplicate_bounces(
        candidates: List[
            Dict[str, Any]
        ],
        minimum_frame_gap: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Merge nearby bounce candidates.

        Keeps the highest-scoring candidate within
        a small temporal neighborhood.
        """

        if not candidates:
            return []

        ordered = sorted(
            candidates,
            key=lambda item: item[
                "frame_number"
            ],
        )

        groups: List[
            List[Dict[str, Any]]
        ] = []

        current_group = [
            ordered[0]
        ]

        for candidate in ordered[
            1:
        ]:
            previous = (
                current_group[-1]
            )

            frame_gap = (
                int(
                    candidate[
                        "frame_number"
                    ]
                )
                - int(
                    previous[
                        "frame_number"
                    ]
                )
            )

            if (
                frame_gap
                <= minimum_frame_gap
            ):
                current_group.append(
                    candidate
                )

            else:
                groups.append(
                    current_group
                )

                current_group = [
                    candidate
                ]

        groups.append(
            current_group
        )

        deduplicated = []

        for group in groups:
            best = max(
                group,
                key=lambda item: item[
                    "bounce_score"
                ],
            )

            deduplicated.append(
                best
            )

        return deduplicated

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self,
    ) -> Dict[str, Any]:
        deduplicated = (
            self.deduplicate_bounces(
                self.bounce_candidates
            )
        )

        return {
            "motion_windows": len(
                self.motion_history
            ),

            "raw_bounce_candidates": len(
                self.bounce_candidates
            ),

            "deduplicated_bounces": len(
                deduplicated
            ),

            "bounce_events": (
                deduplicated
            ),
        }

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        self.motion_history.clear()

        self.bounce_candidates.clear()