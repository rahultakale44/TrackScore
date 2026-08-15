from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.app.vision.court_homography import (
    CourtHomography,
    CourtHomographyError,
)


class BallSpeedAnalysisError(Exception):
    """Raised when ball-speed analysis fails."""


@dataclass
class BallSpeedConfig:
    """
    Configuration for real-world tennis ball speed estimation.
    """

    minimum_time_delta: float = 0.001
    minimum_distance_meters: float = 0.01

    smoothing_window: int = 5

    maximum_reasonable_speed_kmh: float = 300.0

    ignore_predicted_points_for_speed: bool = True


class BallSpeedAnalyzer:
    """
    Converts tracked tennis-ball pixel positions into
    real-world court-plane motion metrics.

    Metrics:
    - court coordinates
    - frame-to-frame distance
    - total distance
    - estimated current speed
    - smoothed speed
    - average speed
    - peak speed
    """

    def __init__(
        self,
        homography: CourtHomography,
        config: BallSpeedConfig | None = None,
    ):
        if homography is None:
            raise BallSpeedAnalysisError(
                "CourtHomography instance is required."
            )

        if not homography.is_calibrated():
            raise BallSpeedAnalysisError(
                "Court homography must be calibrated first."
            )

        self.homography = homography

        self.config = (
            config
            if config is not None
            else BallSpeedConfig()
        )

        self._validate_config()

        self.reset()

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if config.minimum_time_delta <= 0:
            raise BallSpeedAnalysisError(
                "minimum_time_delta must be greater than zero."
            )

        if config.minimum_distance_meters < 0:
            raise BallSpeedAnalysisError(
                "minimum_distance_meters cannot be negative."
            )

        if config.smoothing_window <= 0:
            raise BallSpeedAnalysisError(
                "smoothing_window must be greater than zero."
            )

        if (
            config.maximum_reasonable_speed_kmh
            <= 0
        ):
            raise BallSpeedAnalysisError(
                "maximum_reasonable_speed_kmh "
                "must be greater than zero."
            )

    # ============================================================
    # POINT VALIDATION
    # ============================================================

    @staticmethod
    def validate_track_point(
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
                raise BallSpeedAnalysisError(
                    f"Track point missing '{field}'."
                )

    # ============================================================
    # DISTANCE
    # ============================================================

    @staticmethod
    def calculate_distance(
        point_a: Sequence[float],
        point_b: Sequence[float],
    ) -> float:
        if len(point_a) != 2:
            raise BallSpeedAnalysisError(
                "point_a must contain x and y."
            )

        if len(point_b) != 2:
            raise BallSpeedAnalysisError(
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

    # ============================================================
    # SPEED
    # ============================================================

    @staticmethod
    def meters_per_second_to_kmh(
        speed_mps: float,
    ) -> float:
        return float(
            speed_mps * 3.6
        )

    def calculate_speed(
        self,
        distance_meters: float,
        delta_time_seconds: float,
    ) -> float:
        if (
            delta_time_seconds
            < self.config.minimum_time_delta
        ):
            return 0.0

        if (
            distance_meters
            < self.config.minimum_distance_meters
        ):
            return 0.0

        speed_mps = (
            distance_meters
            / delta_time_seconds
        )

        speed_kmh = (
            self.meters_per_second_to_kmh(
                speed_mps
            )
        )

        if (
            speed_kmh
            > self.config.maximum_reasonable_speed_kmh
        ):
            return 0.0

        return round(
            speed_kmh,
            3,
        )

    # ============================================================
    # PIXEL -> COURT
    # ============================================================

    def pixel_to_court(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        try:
            court_x, court_y = (
                self.homography.image_to_court(
                    (
                        float(x),
                        float(y),
                    )
                )
            )

        except CourtHomographyError as error:
            raise BallSpeedAnalysisError(
                f"Unable to map ball position: {error}"
            ) from error

        return (
            float(court_x),
            float(court_y),
        )

    # ============================================================
    # SMOOTHING
    # ============================================================

    def _smoothed_speed(
        self,
    ) -> float:
        if not self.speed_history:
            return 0.0

        values = self.speed_history[
            -self.config.smoothing_window:
        ]

        return round(
            float(
                np.mean(values)
            ),
            3,
        )

    # ============================================================
    # ANALYSE ONE POINT
    # ============================================================

    def analyse_point(
        self,
        point: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.validate_track_point(
            point
        )

        frame_number = int(
            point[
                "frame_number"
            ]
        )

        timestamp = float(
            point[
                "timestamp_seconds"
            ]
        )

        predicted = bool(
            point.get(
                "predicted",
                False,
            )
        )

        court_x, court_y = (
            self.pixel_to_court(
                point["x"],
                point["y"],
            )
        )

        current_position = (
            court_x,
            court_y,
        )

        frame_distance = 0.0
        current_speed = 0.0

        can_measure_speed = True

        if (
            predicted
            and self.config
            .ignore_predicted_points_for_speed
        ):
            can_measure_speed = False

        if (
            self.previous_position is not None
            and self.previous_timestamp is not None
            and can_measure_speed
        ):
            delta_time = (
                timestamp
                - self.previous_timestamp
            )

            if delta_time > 0:
                frame_distance = (
                    self.calculate_distance(
                        self.previous_position,
                        current_position,
                    )
                )

                current_speed = (
                    self.calculate_speed(
                        frame_distance,
                        delta_time,
                    )
                )

                if current_speed > 0:
                    self.total_distance_meters += (
                        frame_distance
                    )

                    self.speed_history.append(
                        current_speed
                    )

                    self.peak_speed_kmh = max(
                        self.peak_speed_kmh,
                        current_speed,
                    )

                    self.valid_speed_samples += 1

        if not predicted:
            self.previous_position = (
                current_position
            )

            self.previous_timestamp = (
                timestamp
            )

        smoothed_speed = (
            self._smoothed_speed()
        )

        average_speed = (
            round(
                float(
                    np.mean(
                        self.speed_history
                    )
                ),
                3,
            )
            if self.speed_history
            else 0.0
        )

        result = {
            "frame_number": (
                frame_number
            ),

            "timestamp_seconds": round(
                timestamp,
                3,
            ),

            "predicted": predicted,

            "pixel_position": {
                "x": round(
                    float(
                        point["x"]
                    ),
                    3,
                ),
                "y": round(
                    float(
                        point["y"]
                    ),
                    3,
                ),
            },

            "court_position": {
                "x_meters": round(
                    court_x,
                    4,
                ),
                "y_meters": round(
                    court_y,
                    4,
                ),
            },

            "frame_distance_meters": round(
                frame_distance,
                4,
            ),

            "current_speed_kmh": round(
                current_speed,
                3,
            ),

            "smoothed_speed_kmh": (
                smoothed_speed
            ),

            "average_speed_kmh": (
                average_speed
            ),

            "peak_speed_kmh": round(
                self.peak_speed_kmh,
                3,
            ),

            "total_distance_meters": round(
                self.total_distance_meters,
                3,
            ),
        }

        self.real_trajectory.append(
            result
        )

        return result

    # ============================================================
    # ANALYSE MULTIPLE POINTS
    # ============================================================

    def analyse_trajectory(
        self,
        trajectory: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        results = []

        for point in trajectory:
            results.append(
                self.analyse_point(
                    point
                )
            )

        return results

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self,
    ) -> Dict[str, Any]:
        average_speed = (
            float(
                np.mean(
                    self.speed_history
                )
            )
            if self.speed_history
            else 0.0
        )

        return {
            "trajectory_points": len(
                self.real_trajectory
            ),

            "valid_speed_samples": (
                self.valid_speed_samples
            ),

            "total_distance_meters": round(
                self.total_distance_meters,
                3,
            ),

            "average_speed_kmh": round(
                average_speed,
                3,
            ),

            "peak_speed_kmh": round(
                self.peak_speed_kmh,
                3,
            ),

            "latest_smoothed_speed_kmh": (
                self._smoothed_speed()
            ),

            "speed_type": (
                "court-plane estimated speed"
            ),
        }

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        self.previous_position: Optional[
            tuple[float, float]
        ] = None

        self.previous_timestamp: Optional[
            float
        ] = None

        self.total_distance_meters: float = 0.0

        self.peak_speed_kmh: float = 0.0

        self.valid_speed_samples: int = 0

        self.speed_history: List[
            float
        ] = []

        self.real_trajectory: List[
            Dict[str, Any]
        ] = []