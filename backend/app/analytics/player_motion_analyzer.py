from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.app.vision.court_homography import (
    CourtHomography,
    CourtHomographyError,
)


class PlayerMotionAnalysisError(Exception):
    """Raised when player motion analytics cannot be calculated."""


@dataclass
class PlayerMotionConfig:
    """
    Configuration for player movement analytics.
    """

    minimum_time_delta: float = 0.001

    maximum_reasonable_speed_kmh: float = 40.0

    smoothing_window: int = 5

    minimum_movement_meters: float = 0.01


class PlayerMotionAnalyzer:
    """
    Converts tracked player positions into real-world
    tennis movement analytics.

    Metrics:
        - Court X/Y position in meters
        - Court zone
        - Frame-to-frame movement
        - Total distance travelled
        - Current speed
        - Smoothed speed
        - Peak speed
        - Average speed
    """

    def __init__(
        self,
        homography: CourtHomography,
        config: PlayerMotionConfig | None = None,
    ):
        if homography is None:
            raise PlayerMotionAnalysisError(
                "CourtHomography instance is required."
            )

        if not homography.is_calibrated():
            raise PlayerMotionAnalysisError(
                "Court homography must be calibrated "
                "before analysing player movement."
            )

        self.homography = homography

        self.config = (
            config
            if config is not None
            else PlayerMotionConfig()
        )

        self._validate_config()

        self.player_state: Dict[
            str,
            Dict[str, Any],
        ] = {}

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if config.minimum_time_delta <= 0:
            raise PlayerMotionAnalysisError(
                "minimum_time_delta must be greater than zero."
            )

        if config.maximum_reasonable_speed_kmh <= 0:
            raise PlayerMotionAnalysisError(
                "maximum_reasonable_speed_kmh must be greater than zero."
            )

        if config.smoothing_window <= 0:
            raise PlayerMotionAnalysisError(
                "smoothing_window must be greater than zero."
            )

        if config.minimum_movement_meters < 0:
            raise PlayerMotionAnalysisError(
                "minimum_movement_meters cannot be negative."
            )

    # ============================================================
    # PLAYER VALIDATION
    # ============================================================

    @staticmethod
    def _validate_player(
        player: Dict[str, Any],
    ) -> None:
        required_fields = [
            "player_label",
            "foot_point",
            "timestamp_seconds",
            "frame_number",
        ]

        for field in required_fields:
            if field not in player:
                raise PlayerMotionAnalysisError(
                    f"Player record is missing '{field}'."
                )

        foot_point = player[
            "foot_point"
        ]

        if (
            "x" not in foot_point
            or "y" not in foot_point
        ):
            raise PlayerMotionAnalysisError(
                "Player foot_point must contain x and y."
            )

    # ============================================================
    # COORDINATE TRANSFORMATION
    # ============================================================

    def player_pixel_to_court(
        self,
        player: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Convert player's foot pixel position
        into real-world tennis court coordinates.
        """

        self._validate_player(
            player
        )

        foot = player[
            "foot_point"
        ]

        try:
            court_x, court_y = (
                self.homography.image_to_court(
                    (
                        float(
                            foot["x"]
                        ),
                        float(
                            foot["y"]
                        ),
                    )
                )
            )

        except CourtHomographyError as error:
            raise PlayerMotionAnalysisError(
                f"Unable to transform player position: {error}"
            ) from error

        return (
            court_x,
            court_y,
        )

    # ============================================================
    # DISTANCE
    # ============================================================

    @staticmethod
    def calculate_distance(
        point_a: Sequence[float],
        point_b: Sequence[float],
    ) -> float:
        """
        Euclidean distance between two court positions.
        """

        if len(point_a) != 2:
            raise PlayerMotionAnalysisError(
                "point_a must contain x and y."
            )

        if len(point_b) != 2:
            raise PlayerMotionAnalysisError(
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

        return round(
            float(
                hypot(
                    dx,
                    dy,
                )
            ),
            4,
        )

    # ============================================================
    # SPEED
    # ============================================================

    @staticmethod
    def meters_per_second_to_kmh(
        speed_mps: float,
    ) -> float:
        return round(
            float(speed_mps) * 3.6,
            3,
        )

    def calculate_speed(
        self,
        distance_meters: float,
        time_delta_seconds: float,
    ) -> float:
        """
        Calculate movement speed in km/h.
        """

        if (
            time_delta_seconds
            < self.config.minimum_time_delta
        ):
            return 0.0

        if (
            distance_meters
            < self.config.minimum_movement_meters
        ):
            return 0.0

        speed_mps = (
            distance_meters
            / time_delta_seconds
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
    # PLAYER STATE
    # ============================================================

    def _create_player_state(
        self,
        label: str,
    ) -> Dict[str, Any]:
        return {
            "player_label": label,
            "previous_position": None,
            "previous_timestamp": None,
            "total_distance_meters": 0.0,
            "peak_speed_kmh": 0.0,
            "speed_history": [],
            "valid_samples": 0,
        }

    def _get_player_state(
        self,
        label: str,
    ) -> Dict[str, Any]:
        if label not in self.player_state:
            self.player_state[
                label
            ] = self._create_player_state(
                label
            )

        return self.player_state[
            label
        ]

    # ============================================================
    # SPEED SMOOTHING
    # ============================================================

    def _smoothed_speed(
        self,
        speed_history: List[float],
    ) -> float:
        if not speed_history:
            return 0.0

        values = speed_history[
            -self.config.smoothing_window:
        ]

        return round(
            float(
                np.mean(values)
            ),
            3,
        )

    # ============================================================
    # SINGLE PLAYER
    # ============================================================

    def analyse_player(
        self,
        player: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyse one tracked player in the current frame.
        """

        self._validate_player(
            player
        )

        label = str(
            player[
                "player_label"
            ]
        )

        timestamp = float(
            player[
                "timestamp_seconds"
            ]
        )

        frame_number = int(
            player[
                "frame_number"
            ]
        )

        court_x, court_y = (
            self.player_pixel_to_court(
                player
            )
        )

        position = (
            court_x,
            court_y,
        )

        state = self._get_player_state(
            label
        )

        movement_distance = 0.0
        current_speed = 0.0

        previous_position = state[
            "previous_position"
        ]

        previous_timestamp = state[
            "previous_timestamp"
        ]

        if (
            previous_position is not None
            and previous_timestamp is not None
        ):
            time_delta = (
                timestamp
                - float(
                    previous_timestamp
                )
            )

            if time_delta > 0:
                movement_distance = (
                    self.calculate_distance(
                        previous_position,
                        position,
                    )
                )

                current_speed = (
                    self.calculate_speed(
                        movement_distance,
                        time_delta,
                    )
                )

                if current_speed > 0:
                    state[
                        "total_distance_meters"
                    ] += movement_distance

                    state[
                        "speed_history"
                    ].append(
                        current_speed
                    )

                    state[
                        "valid_samples"
                    ] += 1

                    state[
                        "peak_speed_kmh"
                    ] = max(
                        float(
                            state[
                                "peak_speed_kmh"
                            ]
                        ),
                        current_speed,
                    )

        state[
            "previous_position"
        ] = position

        state[
            "previous_timestamp"
        ] = timestamp

        smoothed_speed = (
            self._smoothed_speed(
                state[
                    "speed_history"
                ]
            )
        )

        if state[
            "speed_history"
        ]:
            average_speed = round(
                float(
                    np.mean(
                        state[
                            "speed_history"
                        ]
                    )
                ),
                3,
            )

        else:
            average_speed = 0.0

        calibration = (
            self.homography
            .get_calibration_summary()
        )

        width = float(
            calibration[
                "court_width_meters"
            ]
        )

        length = float(
            calibration[
                "court_length_meters"
            ]
        )

        inside_court = (
            0.0
            <= court_x
            <= width
            and
            0.0
            <= court_y
            <= length
        )

        if inside_court:
            zone = (
                self.homography.geometry
                .classify_court_zone(
                    court_x,
                    court_y,
                    calibration[
                        "court_type"
                    ],
                )
            )

        else:
            zone = "out"

        result = dict(
            player
        )

        result[
            "court_position"
        ] = {
            "x_meters": round(
                court_x,
                3,
            ),
            "y_meters": round(
                court_y,
                3,
            ),
            "inside_court": (
                inside_court
            ),
            "zone": zone,
        }

        result[
            "movement"
        ] = {
            "frame_distance_meters": round(
                movement_distance,
                4,
            ),
            "total_distance_meters": round(
                float(
                    state[
                        "total_distance_meters"
                    ]
                ),
                3,
            ),
            "current_speed_kmh": round(
                current_speed,
                3,
            ),
            "smoothed_speed_kmh": round(
                smoothed_speed,
                3,
            ),
            "average_speed_kmh": round(
                average_speed,
                3,
            ),
            "peak_speed_kmh": round(
                float(
                    state[
                        "peak_speed_kmh"
                    ]
                ),
                3,
            ),
        }

        return result

    # ============================================================
    # MULTIPLE PLAYERS
    # ============================================================

    def analyse_players(
        self,
        players: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        analysed = []

        for player in players:
            analysed.append(
                self.analyse_player(
                    player
                )
            )

        return analysed

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self,
    ) -> Dict[str, Any]:
        summary = {}

        for label, state in (
            self.player_state.items()
        ):
            history = state[
                "speed_history"
            ]

            average_speed = (
                float(
                    np.mean(history)
                )
                if history
                else 0.0
            )

            summary[
                label
            ] = {
                "total_distance_meters": round(
                    float(
                        state[
                            "total_distance_meters"
                        ]
                    ),
                    3,
                ),
                "peak_speed_kmh": round(
                    float(
                        state[
                            "peak_speed_kmh"
                        ]
                    ),
                    3,
                ),
                "average_speed_kmh": round(
                    average_speed,
                    3,
                ),
                "valid_samples": int(
                    state[
                        "valid_samples"
                    ]
                ),
            }

        return summary

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        self.player_state.clear()