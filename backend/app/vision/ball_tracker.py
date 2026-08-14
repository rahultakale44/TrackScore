from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ball_detector import (
    BallDetectionError,
    BallDetector,
    BallDetectorConfig,
)


class BallTrackingError(Exception):
    """Raised when temporal tennis-ball tracking fails."""


@dataclass
class BallTrackerConfig:
    """
    Configuration for temporal tennis-ball tracking.
    """

    model_path: str = "yolo11n.pt"

    confidence_threshold: float = 0.05

    maximum_match_distance_pixels: float = 180.0

    maximum_missed_frames: int = 8

    trajectory_length: int = 40

    position_smoothing_alpha: float = 0.65

    velocity_smoothing_alpha: float = 0.55

    minimum_time_delta: float = 0.001

    use_prediction: bool = True

    use_color_fallback: bool = True

    fallback_min_area: float = 3.0

    fallback_max_area: float = 700.0

    fallback_min_circularity: float = 0.15

    fallback_yellow_hue_low: int = 18
    fallback_yellow_hue_high: int = 48

    fallback_min_saturation: int = 55
    fallback_min_value: int = 90

    device: Optional[str] = None


class BallTracker:
    """
    Temporal tennis-ball tracker.

    Combines:

        YOLO sports-ball candidates
                +
        previous position
                +
        velocity prediction
                +
        candidate gating
                +
        optional color fallback
                ↓
        stable ball trajectory

    This is still a tracking foundation.

    Later stages will add:
    - learned temporal ball models
    - bounce detection
    - trajectory smoothing
    - court-coordinate conversion
    - speed estimation
    """

    def __init__(
        self,
        config: BallTrackerConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else BallTrackerConfig()
        )

        self._validate_config()

        try:
            detector_config = BallDetectorConfig(
                model_path=self.config.model_path,
                confidence_threshold=(
                    self.config.confidence_threshold
                ),
                device=self.config.device,
            )

            self.detector = BallDetector(
                detector_config
            )

        except BallDetectionError as error:
            raise BallTrackingError(
                f"Unable to initialize ball detector: {error}"
            ) from error

        self.reset()

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if not config.model_path:
            raise BallTrackingError(
                "model_path cannot be empty."
            )

        if not (
            0.0
            < config.confidence_threshold
            <= 1.0
        ):
            raise BallTrackingError(
                "confidence_threshold must be within (0, 1]."
            )

        if (
            config.maximum_match_distance_pixels
            <= 0
        ):
            raise BallTrackingError(
                "maximum_match_distance_pixels "
                "must be greater than zero."
            )

        if config.maximum_missed_frames < 0:
            raise BallTrackingError(
                "maximum_missed_frames cannot be negative."
            )

        if config.trajectory_length <= 0:
            raise BallTrackingError(
                "trajectory_length must be greater than zero."
            )

        if not (
            0.0
            <= config.position_smoothing_alpha
            <= 1.0
        ):
            raise BallTrackingError(
                "position_smoothing_alpha must be within [0, 1]."
            )

        if not (
            0.0
            <= config.velocity_smoothing_alpha
            <= 1.0
        ):
            raise BallTrackingError(
                "velocity_smoothing_alpha must be within [0, 1]."
            )

        if config.minimum_time_delta <= 0:
            raise BallTrackingError(
                "minimum_time_delta must be greater than zero."
            )

        if config.fallback_min_area < 0:
            raise BallTrackingError(
                "fallback_min_area cannot be negative."
            )

        if (
            config.fallback_max_area
            <= config.fallback_min_area
        ):
            raise BallTrackingError(
                "fallback_max_area must be larger "
                "than fallback_min_area."
            )

    # ============================================================
    # FRAME VALIDATION
    # ============================================================

    @staticmethod
    def validate_frame(
        frame: np.ndarray,
    ) -> None:
        if frame is None:
            raise BallTrackingError(
                "Frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise BallTrackingError(
                "Frame must be a NumPy array."
            )

        if frame.size == 0:
            raise BallTrackingError(
                "Frame cannot be empty."
            )

        if frame.ndim != 3:
            raise BallTrackingError(
                "Frame must have height, width, and channels."
            )

        if frame.shape[2] != 3:
            raise BallTrackingError(
                "Frame must contain exactly 3 channels."
            )

    # ============================================================
    # BASIC GEOMETRY
    # ============================================================

    @staticmethod
    def calculate_distance(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
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

    # ============================================================
    # POSITION PREDICTION
    # ============================================================

    def predict_position(
        self,
        timestamp_seconds: float,
    ) -> Optional[
        Tuple[float, float]
    ]:
        """
        Predict ball position using last known velocity.
        """

        if self.last_position is None:
            return None

        if self.last_timestamp is None:
            return self.last_position

        if not self.config.use_prediction:
            return self.last_position

        delta_time = (
            float(timestamp_seconds)
            - float(self.last_timestamp)
        )

        if delta_time <= 0:
            return self.last_position

        predicted_x = (
            self.last_position[0]
            + self.velocity[0]
            * delta_time
        )

        predicted_y = (
            self.last_position[1]
            + self.velocity[1]
            * delta_time
        )

        return (
            float(predicted_x),
            float(predicted_y),
        )

    # ============================================================
    # YOLO CANDIDATE SELECTION
    # ============================================================

    def select_temporal_candidate(
        self,
        candidates: List[
            Dict[str, Any]
        ],
        predicted_position: Optional[
            Tuple[float, float]
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Choose the candidate most compatible with temporal motion.
        """

        if not candidates:
            return None

        if predicted_position is None:
            return candidates[0]

        valid_candidates = []

        for candidate in candidates:
            center = candidate[
                "center"
            ]

            candidate_position = (
                float(center["x"]),
                float(center["y"]),
            )

            distance = (
                self.calculate_distance(
                    predicted_position,
                    candidate_position,
                )
            )

            if (
                distance
                > self.config.maximum_match_distance_pixels
            ):
                continue

            item = dict(
                candidate
            )

            item[
                "prediction_distance_pixels"
            ] = round(
                distance,
                3,
            )

            ranking_score = float(
                item.get(
                    "ranking_score",
                    item.get(
                        "confidence",
                        0.0,
                    ),
                )
            )

            distance_score = max(
                0.0,
                1.0
                - (
                    distance
                    / self.config.maximum_match_distance_pixels
                ),
            )

            temporal_score = (
                ranking_score * 0.55
                + distance_score * 0.45
            )

            item[
                "temporal_score"
            ] = round(
                temporal_score,
                4,
            )

            valid_candidates.append(
                item
            )

        if not valid_candidates:
            return None

        valid_candidates.sort(
            key=lambda item: item[
                "temporal_score"
            ],
            reverse=True,
        )

        return valid_candidates[0]

    # ============================================================
    # COLOR-BASED FALLBACK
    # ============================================================

    def find_color_fallback_candidates(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Search for small yellow-green blobs.

        This is only a fallback when YOLO misses the ball.
        """

        self.validate_frame(frame)

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        lower = np.array(
            [
                self.config.fallback_yellow_hue_low,
                self.config.fallback_min_saturation,
                self.config.fallback_min_value,
            ],
            dtype=np.uint8,
        )

        upper = np.array(
            [
                self.config.fallback_yellow_hue_high,
                255,
                255,
            ],
            dtype=np.uint8,
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper,
        )

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:
            area = float(
                cv2.contourArea(
                    contour
                )
            )

            if (
                area
                < self.config.fallback_min_area
            ):
                continue

            if (
                area
                > self.config.fallback_max_area
            ):
                continue

            perimeter = float(
                cv2.arcLength(
                    contour,
                    True,
                )
            )

            if perimeter <= 0:
                continue

            circularity = (
                4.0
                * np.pi
                * area
                / (
                    perimeter
                    * perimeter
                )
            )

            if (
                circularity
                < self.config.fallback_min_circularity
            ):
                continue

            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            center_x = (
                x + width / 2.0
            )

            center_y = (
                y + height / 2.0
            )

            candidates.append(
                {
                    "source": "color_fallback",

                    "confidence": 0.0,

                    "ranking_score": round(
                        min(
                            float(circularity),
                            1.0,
                        ),
                        4,
                    ),

                    "bbox": {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(
                            x + width
                        ),
                        "y2": float(
                            y + height
                        ),
                        "width": float(
                            width
                        ),
                        "height": float(
                            height
                        ),
                        "area": round(
                            area,
                            2,
                        ),
                    },

                    "center": {
                        "x": round(
                            center_x,
                            2,
                        ),
                        "y": round(
                            center_y,
                            2,
                        ),
                    },

                    "circularity": round(
                        float(circularity),
                        4,
                    ),
                }
            )

        candidates.sort(
            key=lambda item: item[
                "ranking_score"
            ],
            reverse=True,
        )

        return candidates

    # ============================================================
    # POSITION SMOOTHING
    # ============================================================

    def smooth_position(
        self,
        new_position: Tuple[
            float,
            float,
        ],
    ) -> Tuple[float, float]:
        """
        Exponential moving average for ball position.
        """

        if self.last_position is None:
            return new_position

        alpha = (
            self.config.position_smoothing_alpha
        )

        smoothed_x = (
            alpha
            * new_position[0]
            + (
                1.0 - alpha
            )
            * self.last_position[0]
        )

        smoothed_y = (
            alpha
            * new_position[1]
            + (
                1.0 - alpha
            )
            * self.last_position[1]
        )

        return (
            float(smoothed_x),
            float(smoothed_y),
        )

    # ============================================================
    # VELOCITY
    # ============================================================

    def update_velocity(
        self,
        new_position: Tuple[
            float,
            float,
        ],
        timestamp_seconds: float,
    ) -> None:
        """
        Calculate smoothed pixel velocity.
        """

        if self.last_position is None:
            self.velocity = (
                0.0,
                0.0,
            )

            return

        if self.last_timestamp is None:
            self.velocity = (
                0.0,
                0.0,
            )

            return

        delta_time = (
            float(timestamp_seconds)
            - float(self.last_timestamp)
        )

        if (
            delta_time
            < self.config.minimum_time_delta
        ):
            return

        raw_velocity_x = (
            new_position[0]
            - self.last_position[0]
        ) / delta_time

        raw_velocity_y = (
            new_position[1]
            - self.last_position[1]
        ) / delta_time

        alpha = (
            self.config.velocity_smoothing_alpha
        )

        velocity_x = (
            alpha
            * raw_velocity_x
            + (
                1.0 - alpha
            )
            * self.velocity[0]
        )

        velocity_y = (
            alpha
            * raw_velocity_y
            + (
                1.0 - alpha
            )
            * self.velocity[1]
        )

        self.velocity = (
            float(velocity_x),
            float(velocity_y),
        )

    # ============================================================
    # TRAJECTORY HISTORY
    # ============================================================

    def add_to_history(
        self,
        position: Tuple[
            float,
            float,
        ],
        frame_number: int,
        timestamp_seconds: float,
        source: str,
        predicted: bool = False,
    ) -> None:
        self.history.append(
            {
                "frame_number": (
                    int(
                        frame_number
                    )
                ),

                "timestamp_seconds": round(
                    float(
                        timestamp_seconds
                    ),
                    3,
                ),

                "x": round(
                    float(
                        position[0]
                    ),
                    3,
                ),

                "y": round(
                    float(
                        position[1]
                    ),
                    3,
                ),

                "source": source,

                "predicted": (
                    predicted
                ),
            }
        )

        if (
            len(self.history)
            > self.config.trajectory_length
        ):
            self.history = self.history[
                -self.config.trajectory_length:
            ]

    # ============================================================
    # REGISTER DETECTION
    # ============================================================

    def register_detection(
        self,
        candidate: Dict[str, Any],
        frame_number: int,
        timestamp_seconds: float,
        source: str,
    ) -> Dict[str, Any]:
        """
        Register one accepted real ball detection.
        """

        center = candidate[
            "center"
        ]

        raw_position = (
            float(
                center["x"]
            ),
            float(
                center["y"]
            ),
        )

        smoothed_position = (
            self.smooth_position(
                raw_position
            )
        )

        self.update_velocity(
            smoothed_position,
            timestamp_seconds,
        )

        self.last_position = (
            smoothed_position
        )

        self.last_timestamp = float(
            timestamp_seconds
        )

        self.missed_frames = 0

        self.add_to_history(
            position=smoothed_position,
            frame_number=frame_number,
            timestamp_seconds=(
                timestamp_seconds
            ),
            source=source,
            predicted=False,
        )

        result = dict(
            candidate
        )

        result[
            "source"
        ] = source

        result[
            "raw_center"
        ] = {
            "x": round(
                raw_position[0],
                3,
            ),
            "y": round(
                raw_position[1],
                3,
            ),
        }

        result[
            "tracked_center"
        ] = {
            "x": round(
                smoothed_position[0],
                3,
            ),
            "y": round(
                smoothed_position[1],
                3,
            ),
        }

        result[
            "velocity_pixels_per_second"
        ] = {
            "x": round(
                self.velocity[0],
                3,
            ),
            "y": round(
                self.velocity[1],
                3,
            ),
        }

        result[
            "predicted"
        ] = False

        return result

    # ============================================================
    # HANDLE MISSED DETECTION
    # ============================================================

    def handle_missing_detection(
        self,
        frame_number: int,
        timestamp_seconds: float,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Temporarily propagate predicted position when detection
        disappears for a few frames.
        """

        self.missed_frames += 1

        if (
            self.missed_frames
            > self.config.maximum_missed_frames
        ):
            self.last_position = None
            self.last_timestamp = None
            self.velocity = (
                0.0,
                0.0,
            )

            return None

        predicted = (
            self.predict_position(
                timestamp_seconds
            )
        )

        if predicted is None:
            return None

        self.last_position = (
            predicted
        )

        self.last_timestamp = float(
            timestamp_seconds
        )

        self.add_to_history(
            position=predicted,
            frame_number=frame_number,
            timestamp_seconds=(
                timestamp_seconds
            ),
            source="motion_prediction",
            predicted=True,
        )

        return {
            "source": "motion_prediction",

            "tracked_center": {
                "x": round(
                    predicted[0],
                    3,
                ),
                "y": round(
                    predicted[1],
                    3,
                ),
            },

            "velocity_pixels_per_second": {
                "x": round(
                    self.velocity[0],
                    3,
                ),
                "y": round(
                    self.velocity[1],
                    3,
                ),
            },

            "predicted": True,

            "missed_frames": (
                self.missed_frames
            ),
        }

    # ============================================================
    # COMPLETE FRAME PIPELINE
    # ============================================================

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
    ) -> Dict[str, Any]:
        """
        Run temporal ball tracking for one video frame.
        """

        self.validate_frame(
            frame
        )

        predicted_position = (
            self.predict_position(
                timestamp_seconds
            )
        )

        detection_result = (
            self.detector.detect_ball(
                frame
            )
        )

        yolo_candidates = (
            detection_result[
                "candidates"
            ]
        )

        selected_candidate = (
            self.select_temporal_candidate(
                yolo_candidates,
                predicted_position,
            )
        )

        source = "yolo"

        fallback_count = 0

        if (
            selected_candidate is None
            and self.config.use_color_fallback
        ):
            fallback_candidates = (
                self.find_color_fallback_candidates(
                    frame
                )
            )

            fallback_count = len(
                fallback_candidates
            )

            selected_candidate = (
                self.select_temporal_candidate(
                    fallback_candidates,
                    predicted_position,
                )
            )

            source = "color_fallback"

        if selected_candidate is not None:
            tracked_ball = (
                self.register_detection(
                    candidate=(
                        selected_candidate
                    ),
                    frame_number=(
                        frame_number
                    ),
                    timestamp_seconds=(
                        timestamp_seconds
                    ),
                    source=source,
                )
            )

            ball_visible = True

        else:
            tracked_ball = (
                self.handle_missing_detection(
                    frame_number=(
                        frame_number
                    ),
                    timestamp_seconds=(
                        timestamp_seconds
                    ),
                )
            )

            ball_visible = False

        return {
            "frame_number": (
                frame_number
            ),

            "timestamp_seconds": round(
                timestamp_seconds,
                3,
            ),

            "yolo_candidate_count": len(
                yolo_candidates
            ),

            "fallback_candidate_count": (
                fallback_count
            ),

            "ball_visible": (
                ball_visible
            ),

            "track_active": (
                tracked_ball
                is not None
            ),

            "missed_frames": (
                self.missed_frames
            ),

            "predicted_position_before_detection": (
                {
                    "x": round(
                        predicted_position[0],
                        3,
                    ),
                    "y": round(
                        predicted_position[1],
                        3,
                    ),
                }
                if predicted_position
                is not None
                else None
            ),

            "ball": tracked_ball,

            "trajectory": list(
                self.history
            ),
        }

    # ============================================================
    # DRAW TRACKING
    # ============================================================

    def draw_tracking(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
    ) -> np.ndarray:
        """
        Draw tracked ball and recent trajectory.
        """

        self.validate_frame(
            frame
        )

        overlay = frame.copy()

        history = result.get(
            "trajectory",
            [],
        )

        if len(history) >= 2:
            for index in range(
                1,
                len(history),
            ):
                previous = history[
                    index - 1
                ]

                current = history[
                    index
                ]

                point_a = (
                    int(
                        round(
                            previous["x"]
                        )
                    ),
                    int(
                        round(
                            previous["y"]
                        )
                    ),
                )

                point_b = (
                    int(
                        round(
                            current["x"]
                        )
                    ),
                    int(
                        round(
                            current["y"]
                        )
                    ),
                )

                if current[
                    "predicted"
                ]:
                    color = (
                        0,
                        165,
                        255,
                    )

                else:
                    color = (
                        0,
                        255,
                        255,
                    )

                cv2.line(
                    overlay,
                    point_a,
                    point_b,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        ball = result.get(
            "ball"
        )

        if ball is None:
            cv2.putText(
                overlay,
                "BALL TRACK LOST",
                (
                    25,
                    40,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (
                    0,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            return overlay

        tracked_center = ball.get(
            "tracked_center"
        )

        if tracked_center is None:
            return overlay

        center_x = int(
            round(
                tracked_center["x"]
            )
        )

        center_y = int(
            round(
                tracked_center["y"]
            )
        )

        predicted = bool(
            ball.get(
                "predicted",
                False,
            )
        )

        if predicted:
            color = (
                0,
                165,
                255,
            )

            status = (
                "BALL PREDICTED"
            )

        else:
            color = (
                0,
                255,
                0,
            )

            status = (
                "BALL TRACKED"
            )

        cv2.circle(
            overlay,
            (
                center_x,
                center_y,
            ),
            8,
            color,
            2,
        )

        cv2.circle(
            overlay,
            (
                center_x,
                center_y,
            ),
            3,
            color,
            -1,
        )

        cv2.putText(
            overlay,
            status,
            (
                max(
                    center_x + 12,
                    10,
                ),
                max(
                    center_y - 10,
                    25,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        return overlay

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self,
    ) -> Dict[str, Any]:
        detected_points = [
            item
            for item in self.history
            if not item[
                "predicted"
            ]
        ]

        predicted_points = [
            item
            for item in self.history
            if item[
                "predicted"
            ]
        ]

        return {
            "trajectory_points": len(
                self.history
            ),

            "detected_points": len(
                detected_points
            ),

            "predicted_points": len(
                predicted_points
            ),

            "current_velocity_pixels_per_second": {
                "x": round(
                    self.velocity[0],
                    3,
                ),

                "y": round(
                    self.velocity[1],
                    3,
                ),
            },

            "missed_frames": (
                self.missed_frames
            ),

            "track_active": (
                self.last_position
                is not None
            ),
        }

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        """
        Reset ball-tracking state.
        """

        self.last_position: Optional[
            Tuple[float, float]
        ] = None

        self.last_timestamp: Optional[
            float
        ] = None

        self.velocity: Tuple[
            float,
            float,
        ] = (
            0.0,
            0.0,
        )

        self.missed_frames: int = 0

        self.history: List[
            Dict[str, Any]
        ] = []