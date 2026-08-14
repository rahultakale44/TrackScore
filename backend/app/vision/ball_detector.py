from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class BallDetectionError(Exception):
    """Raised when tennis ball detection fails."""


@dataclass
class BallDetectorConfig:
    """
    Configuration for tennis ball detection.
    """

    model_path: str = "yolo11n.pt"

    confidence_threshold: float = 0.05
    iou_threshold: float = 0.40

    sports_ball_class_id: int = 32

    minimum_box_area: float = 2.0
    maximum_box_area_ratio: float = 0.01

    minimum_aspect_ratio: float = 0.35
    maximum_aspect_ratio: float = 2.80

    use_color_score: bool = True

    yellow_hue_low: int = 20
    yellow_hue_high: int = 45

    minimum_saturation: int = 60
    minimum_value: int = 80

    device: Optional[str] = None


class BallDetector:
    """
    Detects tennis-ball candidates from video frames.

    Pipeline:

        Frame
          ↓
        YOLO sports-ball detection
          ↓
        Tiny-object filtering
          ↓
        Shape filtering
          ↓
        Tennis-color scoring
          ↓
        Candidate ranking
          ↓
        Best ball candidate
    """

    def __init__(
        self,
        config: BallDetectorConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else BallDetectorConfig()
        )

        self._validate_config()

        try:
            self.model = YOLO(
                self.config.model_path
            )

        except Exception as error:
            raise BallDetectionError(
                f"Unable to load YOLO model "
                f"'{self.config.model_path}': {error}"
            ) from error

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if not config.model_path:
            raise BallDetectionError(
                "model_path cannot be empty."
            )

        if not 0.0 < config.confidence_threshold <= 1.0:
            raise BallDetectionError(
                "confidence_threshold must be within (0, 1]."
            )

        if not 0.0 < config.iou_threshold <= 1.0:
            raise BallDetectionError(
                "iou_threshold must be within (0, 1]."
            )

        if config.minimum_box_area < 0:
            raise BallDetectionError(
                "minimum_box_area cannot be negative."
            )

        if not 0.0 < config.maximum_box_area_ratio <= 1.0:
            raise BallDetectionError(
                "maximum_box_area_ratio must be within (0, 1]."
            )

        if config.minimum_aspect_ratio <= 0:
            raise BallDetectionError(
                "minimum_aspect_ratio must be greater than zero."
            )

        if (
            config.maximum_aspect_ratio
            <= config.minimum_aspect_ratio
        ):
            raise BallDetectionError(
                "maximum_aspect_ratio must be greater "
                "than minimum_aspect_ratio."
            )

    # ============================================================
    # FRAME VALIDATION
    # ============================================================

    @staticmethod
    def validate_frame(
        frame: np.ndarray,
    ) -> None:
        if frame is None:
            raise BallDetectionError(
                "Frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise BallDetectionError(
                "Frame must be a NumPy array."
            )

        if frame.size == 0:
            raise BallDetectionError(
                "Frame cannot be empty."
            )

        if frame.ndim != 3:
            raise BallDetectionError(
                "Frame must have height, width, and channels."
            )

        if frame.shape[2] != 3:
            raise BallDetectionError(
                "Frame must contain exactly 3 channels."
            )

    # ============================================================
    # RAW DETECTION
    # ============================================================

    def detect_raw_candidates(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect general sports-ball candidates.
        """

        self.validate_frame(frame)

        try:
            results = self.model.predict(
                source=frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=[
                    self.config.sports_ball_class_id
                ],
                device=self.config.device,
                verbose=False,
            )

        except Exception as error:
            raise BallDetectionError(
                f"Ball detection inference failed: {error}"
            ) from error

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes

        if len(boxes) == 0:
            return []

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        frame_area = float(
            frame_width
            * frame_height
        )

        candidates: List[
            Dict[str, Any]
        ] = []

        for index in range(
            len(xyxy)
        ):
            x1, y1, x2, y2 = [
                float(value)
                for value in xyxy[index]
            ]

            confidence = float(
                confidences[index]
            )

            width = max(
                0.0,
                x2 - x1,
            )

            height = max(
                0.0,
                y2 - y1,
            )

            area = (
                width
                * height
            )

            if height > 0:
                aspect_ratio = (
                    width
                    / height
                )
            else:
                aspect_ratio = 0.0

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            candidates.append(
                {
                    "candidate_id": index + 1,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "width": round(
                            width,
                            2,
                        ),
                        "height": round(
                            height,
                            2,
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
                    "aspect_ratio": round(
                        aspect_ratio,
                        4,
                    ),
                    "area_ratio": round(
                        area / frame_area,
                        8,
                    ),
                }
            )

        return candidates

    # ============================================================
    # SIZE / SHAPE FILTER
    # ============================================================

    def filter_candidates(
        self,
        candidates: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        filtered = []

        for candidate in candidates:
            area = float(
                candidate[
                    "bbox"
                ]["area"]
            )

            area_ratio = float(
                candidate[
                    "area_ratio"
                ]
            )

            aspect_ratio = float(
                candidate[
                    "aspect_ratio"
                ]
            )

            if (
                area
                < self.config.minimum_box_area
            ):
                continue

            if (
                area_ratio
                > self.config.maximum_box_area_ratio
            ):
                continue

            if (
                aspect_ratio
                < self.config.minimum_aspect_ratio
            ):
                continue

            if (
                aspect_ratio
                > self.config.maximum_aspect_ratio
            ):
                continue

            filtered.append(
                candidate
            )

        return filtered

    # ============================================================
    # COLOR SCORE
    # ============================================================

    def calculate_color_score(
        self,
        frame: np.ndarray,
        candidate: Dict[str, Any],
    ) -> float:
        """
        Estimate how tennis-ball-like the candidate crop is
        using HSV color distribution.
        """

        self.validate_frame(frame)

        bbox = candidate[
            "bbox"
        ]

        frame_height, frame_width = (
            frame.shape[:2]
        )

        x1 = max(
            0,
            int(
                round(
                    bbox["x1"]
                )
            ),
        )

        y1 = max(
            0,
            int(
                round(
                    bbox["y1"]
                )
            ),
        )

        x2 = min(
            frame_width,
            int(
                round(
                    bbox["x2"]
                )
            ),
        )

        y2 = min(
            frame_height,
            int(
                round(
                    bbox["y2"]
                )
            ),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            return 0.0

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return 0.0

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV,
        )

        lower = np.array(
            [
                self.config.yellow_hue_low,
                self.config.minimum_saturation,
                self.config.minimum_value,
            ],
            dtype=np.uint8,
        )

        upper = np.array(
            [
                self.config.yellow_hue_high,
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

        yellow_pixels = int(
            np.count_nonzero(
                mask
            )
        )

        total_pixels = int(
            mask.size
        )

        if total_pixels == 0:
            return 0.0

        score = (
            yellow_pixels
            / total_pixels
        )

        return round(
            float(score),
            4,
        )

    # ============================================================
    # CANDIDATE RANKING
    # ============================================================

    def rank_candidates(
        self,
        frame: np.ndarray,
        candidates: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        ranked = []

        for candidate in candidates:
            item = dict(
                candidate
            )

            confidence = float(
                candidate[
                    "confidence"
                ]
            )

            if (
                self.config.use_color_score
            ):
                color_score = (
                    self.calculate_color_score(
                        frame,
                        candidate,
                    )
                )
            else:
                color_score = 0.0

            item[
                "color_score"
            ] = color_score

            ranking_score = (
                confidence * 0.75
                + color_score * 0.25
            )

            item[
                "ranking_score"
            ] = round(
                ranking_score,
                4,
            )

            ranked.append(
                item
            )

        ranked.sort(
            key=lambda item: item[
                "ranking_score"
            ],
            reverse=True,
        )

        return ranked

    # ============================================================
    # COMPLETE PIPELINE
    # ============================================================

    def detect_ball(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        raw_candidates = (
            self.detect_raw_candidates(
                frame
            )
        )

        filtered_candidates = (
            self.filter_candidates(
                raw_candidates
            )
        )

        ranked_candidates = (
            self.rank_candidates(
                frame,
                filtered_candidates,
            )
        )

        best_candidate = (
            ranked_candidates[0]
            if ranked_candidates
            else None
        )

        return {
            "raw_candidate_count": len(
                raw_candidates
            ),
            "filtered_candidate_count": len(
                filtered_candidates
            ),
            "ball_detected": (
                best_candidate
                is not None
            ),
            "ball": best_candidate,
            "candidates": ranked_candidates,
        }

    # ============================================================
    # DRAW OVERLAY
    # ============================================================

    @staticmethod
    def draw_detection(
        frame: np.ndarray,
        result: Dict[str, Any],
    ) -> np.ndarray:
        BallDetector.validate_frame(
            frame
        )

        overlay = frame.copy()

        ball = result.get(
            "ball"
        )

        if ball is None:
            cv2.putText(
                overlay,
                "BALL: NOT DETECTED",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            return overlay

        bbox = ball[
            "bbox"
        ]

        x1 = int(
            round(
                bbox["x1"]
            )
        )

        y1 = int(
            round(
                bbox["y1"]
            )
        )

        x2 = int(
            round(
                bbox["x2"]
            )
        )

        y2 = int(
            round(
                bbox["y2"]
            )
        )

        center_x = int(
            round(
                ball[
                    "center"
                ]["x"]
            )
        )

        center_y = int(
            round(
                ball[
                    "center"
                ]["y"]
            )
        )

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.circle(
            overlay,
            (
                center_x,
                center_y,
            ),
            6,
            (0, 0, 255),
            -1,
        )

        label = (
            f"BALL "
            f"{ball['confidence']:.2f} "
            f"| Score "
            f"{ball['ranking_score']:.2f}"
        )

        cv2.putText(
            overlay,
            label,
            (
                x1,
                max(
                    y1 - 10,
                    25,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

    # ============================================================
    # SAVE
    # ============================================================

    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: str,
    ) -> Path:
        if image is None:
            raise BallDetectionError(
                "Cannot save None image."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise BallDetectionError(
                "Image must be NumPy array."
            )

        if image.size == 0:
            raise BallDetectionError(
                "Cannot save empty image."
            )

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output),
            image,
        )

        if not success:
            raise BallDetectionError(
                f"Unable to save image: {output}"
            )

        return output