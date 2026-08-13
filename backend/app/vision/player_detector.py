from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class PlayerDetectionError(Exception):
    """Raised when tennis player detection fails."""


@dataclass
class PlayerDetectorConfig:
    """
    Configuration for tennis player detection.
    """

    model_path: str = "yolo11n.pt"

    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45

    person_class_id: int = 0

    max_players: int = 2

    court_roi_top_ratio: float = 0.15
    court_roi_bottom_ratio: float = 1.00

    minimum_box_height_ratio: float = 0.03

    device: Optional[str] = None


class PlayerDetector:
    """
    Detects tennis players from match frames.

    Pipeline:

        Frame
          ↓
        YOLO Person Detection
          ↓
        Person-only Filtering
          ↓
        Court ROI Filtering
          ↓
        Bounding-box Size Filtering
          ↓
        Candidate Ranking
          ↓
        Top Tennis Players
          ↓
        Player A / Player B Assignment

    Notes:
    - The pretrained detector identifies general people.
    - Tennis-specific filtering is performed afterwards.
    - Persistent player IDs will be added in the tracking stage.
    """

    def __init__(
        self,
        config: PlayerDetectorConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else PlayerDetectorConfig()
        )

        self._validate_config()

        try:
            self.model = YOLO(
                self.config.model_path
            )

        except Exception as error:
            raise PlayerDetectionError(
                f"Unable to load YOLO model "
                f"'{self.config.model_path}': {error}"
            ) from error

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if not config.model_path:
            raise PlayerDetectionError(
                "model_path cannot be empty."
            )

        if not 0.0 < config.confidence_threshold <= 1.0:
            raise PlayerDetectionError(
                "confidence_threshold must be within (0, 1]."
            )

        if not 0.0 < config.iou_threshold <= 1.0:
            raise PlayerDetectionError(
                "iou_threshold must be within (0, 1]."
            )

        if config.max_players <= 0:
            raise PlayerDetectionError(
                "max_players must be greater than zero."
            )

        if not 0.0 <= config.court_roi_top_ratio < 1.0:
            raise PlayerDetectionError(
                "court_roi_top_ratio must be within [0, 1)."
            )

        if not 0.0 < config.court_roi_bottom_ratio <= 1.0:
            raise PlayerDetectionError(
                "court_roi_bottom_ratio must be within (0, 1]."
            )

        if (
            config.court_roi_top_ratio
            >= config.court_roi_bottom_ratio
        ):
            raise PlayerDetectionError(
                "court_roi_top_ratio must be smaller "
                "than court_roi_bottom_ratio."
            )

        if not 0.0 < config.minimum_box_height_ratio <= 1.0:
            raise PlayerDetectionError(
                "minimum_box_height_ratio must be within (0, 1]."
            )

    # ============================================================
    # FRAME VALIDATION
    # ============================================================

    @staticmethod
    def validate_frame(
        frame: np.ndarray,
    ) -> None:
        if frame is None:
            raise PlayerDetectionError(
                "Frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise PlayerDetectionError(
                "Frame must be a NumPy array."
            )

        if frame.size == 0:
            raise PlayerDetectionError(
                "Frame cannot be empty."
            )

        if frame.ndim != 3:
            raise PlayerDetectionError(
                "Frame must have height, width, and channels."
            )

        if frame.shape[2] != 3:
            raise PlayerDetectionError(
                "Frame must contain exactly 3 channels."
            )

    # ============================================================
    # RAW YOLO DETECTION
    # ============================================================

    def detect_people(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect all visible persons in a frame.
        """

        self.validate_frame(frame)

        try:
            results = self.model.predict(
                source=frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=[
                    self.config.person_class_id
                ],
                device=self.config.device,
                verbose=False,
            )

        except Exception as error:
            raise PlayerDetectionError(
                f"YOLO inference failed: {error}"
            ) from error

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes

        if len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        height, width = frame.shape[:2]

        detections: List[
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

            class_id = int(
                class_ids[index]
            )

            box_width = max(
                0.0,
                x2 - x1,
            )

            box_height = max(
                0.0,
                y2 - y1,
            )

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            foot_x = center_x
            foot_y = y2

            detections.append(
                {
                    "detection_id": index + 1,
                    "class_id": class_id,
                    "class_name": "person",
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
                            box_width,
                            2,
                        ),
                        "height": round(
                            box_height,
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
                    "foot_point": {
                        "x": round(
                            foot_x,
                            2,
                        ),
                        "y": round(
                            foot_y,
                            2,
                        ),
                    },
                    "normalized": {
                        "center_x": round(
                            center_x / width,
                            6,
                        ),
                        "center_y": round(
                            center_y / height,
                            6,
                        ),
                        "box_width": round(
                            box_width / width,
                            6,
                        ),
                        "box_height": round(
                            box_height / height,
                            6,
                        ),
                    },
                }
            )

        return detections

    # ============================================================
    # COURT ROI FILTER
    # ============================================================

    def filter_court_people(
        self,
        detections: List[
            Dict[str, Any]
        ],
        frame_shape: Tuple[
            int,
            int,
            int,
        ],
    ) -> List[Dict[str, Any]]:
        """
        Filter out people likely belonging to crowd,
        officials, background, or other broadcast regions.
        """

        frame_height = frame_shape[0]

        top_y = (
            frame_height
            * self.config.court_roi_top_ratio
        )

        bottom_y = (
            frame_height
            * self.config.court_roi_bottom_ratio
        )

        minimum_height = (
            frame_height
            * self.config.minimum_box_height_ratio
        )

        filtered: List[
            Dict[str, Any]
        ] = []

        for detection in detections:
            foot_y = float(
                detection[
                    "foot_point"
                ]["y"]
            )

            box_height = float(
                detection[
                    "bbox"
                ]["height"]
            )

            if foot_y < top_y:
                continue

            if foot_y > bottom_y:
                continue

            if box_height < minimum_height:
                continue

            filtered.append(
                detection
            )

        return filtered

    # ============================================================
    # PLAYER SELECTION
    # ============================================================

    @staticmethod
    def _calculate_player_score(
        detection: Dict[str, Any],
    ) -> float:
        """
        Rank candidate tennis players.

        Larger boxes and stronger confidence receive
        higher scores.
        """

        confidence = float(
            detection["confidence"]
        )

        box_height = float(
            detection["bbox"]["height"]
        )

        box_width = float(
            detection["bbox"]["width"]
        )

        area = (
            box_width
            * box_height
        )

        score = (
            confidence
            * np.sqrt(
                max(
                    area,
                    1.0,
                )
            )
        )

        return float(score)

    def select_players(
        self,
        detections: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        """
        Select the strongest tennis-player candidates.
        """

        ranked = []

        for detection in detections:
            candidate = dict(
                detection
            )

            candidate[
                "selection_score"
            ] = round(
                self._calculate_player_score(
                    detection
                ),
                4,
            )

            ranked.append(
                candidate
            )

        ranked.sort(
            key=lambda item: item[
                "selection_score"
            ],
            reverse=True,
        )

        selected = ranked[
            : self.config.max_players
        ]

        return selected

    # ============================================================
    # PLAYER LABEL ASSIGNMENT
    # ============================================================

    @staticmethod
    def assign_player_labels(
        players: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        """
        Assign Player A and Player B based on vertical
        image location.

        The farther player generally appears higher
        in a standard tennis broadcast view.
        """

        if not players:
            return []

        ordered = sorted(
            players,
            key=lambda item: item[
                "foot_point"
            ]["y"],
        )

        labelled = []

        labels = [
            "Player A",
            "Player B",
        ]

        for index, player in enumerate(
            ordered
        ):
            item = dict(
                player
            )

            if index < len(labels):
                item[
                    "player_label"
                ] = labels[index]

            else:
                item[
                    "player_label"
                ] = (
                    f"Player {index + 1}"
                )

            item[
                "court_side"
            ] = (
                "far"
                if index == 0
                else "near"
            )

            labelled.append(
                item
            )

        return labelled

    # ============================================================
    # COMPLETE PIPELINE
    # ============================================================

    def detect_players(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run complete tennis-player detection.
        """

        self.validate_frame(frame)

        raw_people = self.detect_people(
            frame
        )

        court_people = (
            self.filter_court_people(
                raw_people,
                frame.shape,
            )
        )

        selected_players = (
            self.select_players(
                court_people
            )
        )

        labelled_players = (
            self.assign_player_labels(
                selected_players
            )
        )

        return {
            "raw_person_count": len(
                raw_people
            ),
            "court_candidate_count": len(
                court_people
            ),
            "selected_player_count": len(
                labelled_players
            ),
            "players": labelled_players,
        }

    # ============================================================
    # DRAWING
    # ============================================================

    @staticmethod
    def draw_players(
        frame: np.ndarray,
        players: List[
            Dict[str, Any]
        ],
    ) -> np.ndarray:
        """
        Draw player bounding boxes and labels.
        """

        PlayerDetector.validate_frame(
            frame
        )

        overlay = frame.copy()

        for player in players:
            bbox = player["bbox"]

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

            label = player.get(
                "player_label",
                "Player",
            )

            confidence = float(
                player[
                    "confidence"
                ]
            )

            court_side = player.get(
                "court_side",
                "",
            )

            text = (
                f"{label} | "
                f"{court_side} | "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            text_y = max(
                30,
                y1 - 10,
            )

            cv2.putText(
                overlay,
                text,
                (
                    x1,
                    text_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            foot = player[
                "foot_point"
            ]

            foot_x = int(
                round(
                    foot["x"]
                )
            )

            foot_y = int(
                round(
                    foot["y"]
                )
            )

            cv2.circle(
                overlay,
                (
                    foot_x,
                    foot_y,
                ),
                6,
                (0, 0, 255),
                -1,
            )

        cv2.putText(
            overlay,
            (
                f"TrackScore Players: "
                f"{len(players)}"
            ),
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

    # ============================================================
    # SAVE IMAGE
    # ============================================================

    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: str,
    ) -> Path:
        if image is None:
            raise PlayerDetectionError(
                "Cannot save None image."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise PlayerDetectionError(
                "Image must be a NumPy array."
            )

        if image.size == 0:
            raise PlayerDetectionError(
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
            raise PlayerDetectionError(
                f"Unable to save image: {output}"
            )

        return output