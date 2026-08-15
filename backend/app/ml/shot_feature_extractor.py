from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class ShotFeatureExtractionError(Exception):
    """Raised when rally or shot feature extraction fails."""


@dataclass
class ShotFeatureConfig:
    """
    Configuration for rally and shot segmentation.
    """

    maximum_gap_seconds: float = 1.25

    minimum_rally_points: int = 4

    minimum_shot_points: int = 2

    direction_change_threshold_degrees: float = 35.0

    minimum_motion_pixels: float = 4.0

    maximum_shot_duration_seconds: float = 2.5


class ShotFeatureExtractor:
    """
    Converts ball trajectory data into rally and shot-level ML features.

    Responsibilities:
    - Rally segmentation
    - Shot candidate segmentation
    - Trajectory statistics
    - Direction features
    - Speed features
    - Position features
    """

    def __init__(
        self,
        config: ShotFeatureConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else ShotFeatureConfig()
        )

        self._validate_config()

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if config.maximum_gap_seconds <= 0:
            raise ShotFeatureExtractionError(
                "maximum_gap_seconds must be greater than zero."
            )

        if config.minimum_rally_points < 2:
            raise ShotFeatureExtractionError(
                "minimum_rally_points must be at least 2."
            )

        if config.minimum_shot_points < 2:
            raise ShotFeatureExtractionError(
                "minimum_shot_points must be at least 2."
            )

        if not (
            0.0
            <= config.direction_change_threshold_degrees
            <= 180.0
        ):
            raise ShotFeatureExtractionError(
                "direction_change_threshold_degrees "
                "must be within [0, 180]."
            )

        if config.minimum_motion_pixels < 0:
            raise ShotFeatureExtractionError(
                "minimum_motion_pixels cannot be negative."
            )

        if (
            config.maximum_shot_duration_seconds
            <= 0
        ):
            raise ShotFeatureExtractionError(
                "maximum_shot_duration_seconds "
                "must be greater than zero."
            )

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def validate_point(
        point: Dict[str, Any],
    ) -> None:
        required = [
            "frame_number",
            "timestamp_seconds",
            "x",
            "y",
        ]

        for field in required:
            if field not in point:
                raise ShotFeatureExtractionError(
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
    def calculate_angle(
        point_a: Dict[str, Any],
        point_b: Dict[str, Any],
    ) -> float:
        dx = (
            float(point_b["x"])
            - float(point_a["x"])
        )

        dy = (
            float(point_b["y"])
            - float(point_a["y"])
        )

        angle = degrees(
            atan2(
                dy,
                dx,
            )
        )

        return float(angle)

    @staticmethod
    def normalize_angle_difference(
        angle_a: float,
        angle_b: float,
    ) -> float:
        difference = (
            angle_b
            - angle_a
        )

        while difference > 180:
            difference -= 360

        while difference < -180:
            difference += 360

        return abs(
            float(
                difference
            )
        )

    # ============================================================
    # RALLY SEGMENTATION
    # ============================================================

    def segment_rallies(
        self,
        trajectory: List[
            Dict[str, Any]
        ],
    ) -> List[
        List[Dict[str, Any]]
    ]:
        """
        Split trajectory into rallies based on temporal gaps.
        """

        if not trajectory:
            return []

        ordered = sorted(
            trajectory,
            key=lambda item: item[
                "timestamp_seconds"
            ],
        )

        rallies = []

        current_rally = [
            ordered[0]
        ]

        for point in ordered[
            1:
        ]:
            previous = (
                current_rally[-1]
            )

            time_gap = (
                float(
                    point[
                        "timestamp_seconds"
                    ]
                )
                - float(
                    previous[
                        "timestamp_seconds"
                    ]
                )
            )

            if (
                time_gap
                > self.config.maximum_gap_seconds
            ):
                if (
                    len(current_rally)
                    >= self.config.minimum_rally_points
                ):
                    rallies.append(
                        current_rally
                    )

                current_rally = [
                    point
                ]

            else:
                current_rally.append(
                    point
                )

        if (
            len(current_rally)
            >= self.config.minimum_rally_points
        ):
            rallies.append(
                current_rally
            )

        return rallies

    # ============================================================
    # SHOT SEGMENTATION
    # ============================================================

    def segment_shots(
        self,
        rally: List[
            Dict[str, Any]
        ],
    ) -> List[
        List[Dict[str, Any]]
    ]:
        """
        Estimate shot boundaries using strong trajectory direction change.

        This is heuristic segmentation.
        ML classification comes later.
        """

        if (
            len(rally)
            < self.config.minimum_shot_points
        ):
            return []

        shots = []

        current_shot = [
            rally[0]
        ]

        previous_angle: Optional[
            float
        ] = None

        for index in range(
            1,
            len(rally)
        ):
            previous_point = (
                rally[
                    index - 1
                ]
            )

            current_point = (
                rally[
                    index
                ]
            )

            distance = (
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

            if (
                distance
                < self.config.minimum_motion_pixels
            ):
                current_shot.append(
                    current_point
                )

                continue

            current_angle = (
                self.calculate_angle(
                    previous_point,
                    current_point,
                )
            )

            direction_change = 0.0

            if (
                previous_angle
                is not None
            ):
                direction_change = (
                    self.normalize_angle_difference(
                        previous_angle,
                        current_angle,
                    )
                )

            shot_start_time = float(
                current_shot[
                    0
                ][
                    "timestamp_seconds"
                ]
            )

            shot_duration = (
                float(
                    current_point[
                        "timestamp_seconds"
                    ]
                )
                - shot_start_time
            )

            should_split = (
                direction_change
                >= self.config
                .direction_change_threshold_degrees
                or
                shot_duration
                >= self.config
                .maximum_shot_duration_seconds
            )

            if (
                should_split
                and len(
                    current_shot
                )
                >= self.config.minimum_shot_points
            ):
                shots.append(
                    current_shot
                )

                current_shot = [
                    previous_point,
                    current_point,
                ]

            else:
                current_shot.append(
                    current_point
                )

            previous_angle = (
                current_angle
            )

        if (
            len(current_shot)
            >= self.config.minimum_shot_points
        ):
            shots.append(
                current_shot
            )

        return shots

    # ============================================================
    # FEATURE EXTRACTION
    # ============================================================

    def extract_shot_features(
        self,
        shot: List[
            Dict[str, Any]
        ],
        rally_id: int,
        shot_id: int,
    ) -> Dict[str, Any]:
        """
        Convert one segmented shot into structured ML features.
        """

        if (
            len(shot)
            < self.config.minimum_shot_points
        ):
            raise ShotFeatureExtractionError(
                "Shot does not contain enough points."
            )

        for point in shot:
            self.validate_point(
                point
            )

        start = shot[0]
        end = shot[-1]

        start_time = float(
            start[
                "timestamp_seconds"
            ]
        )

        end_time = float(
            end[
                "timestamp_seconds"
            ]
        )

        duration = max(
            0.0,
            end_time
            - start_time,
        )

        segment_distances = []

        segment_speeds = []

        angles = []

        predicted_count = 0

        for index in range(
            1,
            len(shot)
        ):
            previous = shot[
                index - 1
            ]

            current = shot[
                index
            ]

            distance = (
                self.calculate_distance(
                    (
                        previous["x"],
                        previous["y"],
                    ),
                    (
                        current["x"],
                        current["y"],
                    ),
                )
            )

            segment_distances.append(
                distance
            )

            delta_time = (
                float(
                    current[
                        "timestamp_seconds"
                    ]
                )
                - float(
                    previous[
                        "timestamp_seconds"
                    ]
                )
            )

            if delta_time > 0:
                segment_speeds.append(
                    distance
                    / delta_time
                )

            angles.append(
                self.calculate_angle(
                    previous,
                    current,
                )
            )

        for point in shot:
            if bool(
                point.get(
                    "predicted",
                    False,
                )
            ):
                predicted_count += 1

        total_distance = float(
            sum(
                segment_distances
            )
        )

        displacement = (
            self.calculate_distance(
                (
                    start["x"],
                    start["y"],
                ),
                (
                    end["x"],
                    end["y"],
                ),
            )
        )

        average_speed = (
            float(
                np.mean(
                    segment_speeds
                )
            )
            if segment_speeds
            else 0.0
        )

        maximum_speed = (
            float(
                np.max(
                    segment_speeds
                )
            )
            if segment_speeds
            else 0.0
        )

        minimum_speed = (
            float(
                np.min(
                    segment_speeds
                )
            )
            if segment_speeds
            else 0.0
        )

        average_angle = (
            float(
                np.mean(
                    angles
                )
            )
            if angles
            else 0.0
        )

        direction_changes = []

        for index in range(
            1,
            len(angles)
        ):
            direction_changes.append(
                self.normalize_angle_difference(
                    angles[
                        index - 1
                    ],
                    angles[index],
                )
            )

        mean_direction_change = (
            float(
                np.mean(
                    direction_changes
                )
            )
            if direction_changes
            else 0.0
        )

        max_direction_change = (
            float(
                np.max(
                    direction_changes
                )
            )
            if direction_changes
            else 0.0
        )

        prediction_ratio = (
            predicted_count
            / len(shot)
        )

        return {
            "rally_id": (
                rally_id
            ),

            "shot_id": (
                shot_id
            ),

            "start_frame": int(
                start[
                    "frame_number"
                ]
            ),

            "end_frame": int(
                end[
                    "frame_number"
                ]
            ),

            "start_time_seconds": round(
                start_time,
                3,
            ),

            "end_time_seconds": round(
                end_time,
                3,
            ),

            "duration_seconds": round(
                duration,
                4,
            ),

            "point_count": len(
                shot
            ),

            "start_x": round(
                float(
                    start["x"]
                ),
                3,
            ),

            "start_y": round(
                float(
                    start["y"]
                ),
                3,
            ),

            "end_x": round(
                float(
                    end["x"]
                ),
                3,
            ),

            "end_y": round(
                float(
                    end["y"]
                ),
                3,
            ),

            "delta_x": round(
                float(
                    end["x"]
                )
                - float(
                    start["x"]
                ),
                3,
            ),

            "delta_y": round(
                float(
                    end["y"]
                )
                - float(
                    start["y"]
                ),
                3,
            ),

            "displacement_pixels": round(
                displacement,
                3,
            ),

            "trajectory_distance_pixels": round(
                total_distance,
                3,
            ),

            "average_speed_pixels_per_second": round(
                average_speed,
                3,
            ),

            "maximum_speed_pixels_per_second": round(
                maximum_speed,
                3,
            ),

            "minimum_speed_pixels_per_second": round(
                minimum_speed,
                3,
            ),

            "average_direction_degrees": round(
                average_angle,
                3,
            ),

            "mean_direction_change_degrees": round(
                mean_direction_change,
                3,
            ),

            "max_direction_change_degrees": round(
                max_direction_change,
                3,
            ),

            "predicted_point_ratio": round(
                prediction_ratio,
                4,
            ),
        }

    # ============================================================
    # FULL DATASET
    # ============================================================

    def build_feature_dataset(
        self,
        trajectory: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        rallies = (
            self.segment_rallies(
                trajectory
            )
        )

        features = []

        rally_summary = []

        global_shot_id = 1

        for rally_index, rally in enumerate(
            rallies,
            start=1,
        ):
            shots = (
                self.segment_shots(
                    rally
                )
            )

            rally_summary.append(
                {
                    "rally_id": (
                        rally_index
                    ),

                    "start_time_seconds": round(
                        float(
                            rally[
                                0
                            ][
                                "timestamp_seconds"
                            ]
                        ),
                        3,
                    ),

                    "end_time_seconds": round(
                        float(
                            rally[
                                -1
                            ][
                                "timestamp_seconds"
                            ]
                        ),
                        3,
                    ),

                    "trajectory_points": len(
                        rally
                    ),

                    "shot_count": len(
                        shots
                    ),
                }
            )

            for shot in shots:
                feature = (
                    self.extract_shot_features(
                        shot=shot,
                        rally_id=(
                            rally_index
                        ),
                        shot_id=(
                            global_shot_id
                        ),
                    )
                )

                features.append(
                    feature
                )

                global_shot_id += 1

        return {
            "rally_count": len(
                rallies
            ),

            "shot_count": len(
                features
            ),

            "rallies": (
                rally_summary
            ),

            "shot_features": (
                features
            ),
        }