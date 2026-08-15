from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class PlayerTrackingError(Exception):
    """Raised when tennis player tracking fails."""


@dataclass
class PlayerTrackerConfig:
    """Configuration for tennis player tracking."""

    model_path: str = "yolo11n.pt"

    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45

    tracker_config: str = "bytetrack.yaml"

    person_class_id: int = 0

    max_players: int = 2

    minimum_box_height_ratio: float = 0.03

    court_roi_top_ratio: float = 0.15
    court_roi_bottom_ratio: float = 1.00

    device: Optional[str] = None


class PlayerTracker:
    """
    Tracks tennis players across video frames using
    YOLO + ByteTrack.

    Responsibilities:
    - Person detection
    - Persistent tracking IDs
    - Court-region filtering
    - Player A / Player B assignment
    - Position history
    - Movement trail visualization
    """

    def __init__(
        self,
        config: PlayerTrackerConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else PlayerTrackerConfig()
        )

        self._validate_config()

        try:
            self.model = YOLO(
                self.config.model_path
            )

        except Exception as error:
            raise PlayerTrackingError(
                f"Unable to load YOLO model "
                f"'{self.config.model_path}': {error}"
            ) from error

        self.track_to_player_label: Dict[int, str] = {}

        self.player_history: Dict[
            str,
            List[Dict[str, Any]],
        ] = {
            "Player A": [],
            "Player B": [],
        }

    # ============================================================
    # CONFIGURATION VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if not config.model_path:
            raise PlayerTrackingError(
                "model_path cannot be empty."
            )

        if not (
            0.0
            < config.confidence_threshold
            <= 1.0
        ):
            raise PlayerTrackingError(
                "confidence_threshold must be within (0, 1]."
            )

        if not (
            0.0
            < config.iou_threshold
            <= 1.0
        ):
            raise PlayerTrackingError(
                "iou_threshold must be within (0, 1]."
            )

        if config.max_players <= 0:
            raise PlayerTrackingError(
                "max_players must be greater than zero."
            )

        if not (
            0.0
            <= config.court_roi_top_ratio
            < 1.0
        ):
            raise PlayerTrackingError(
                "court_roi_top_ratio must be within [0, 1)."
            )

        if not (
            0.0
            < config.court_roi_bottom_ratio
            <= 1.0
        ):
            raise PlayerTrackingError(
                "court_roi_bottom_ratio must be within (0, 1]."
            )

        if (
            config.court_roi_top_ratio
            >= config.court_roi_bottom_ratio
        ):
            raise PlayerTrackingError(
                "court_roi_top_ratio must be smaller "
                "than court_roi_bottom_ratio."
            )

        if not (
            0.0
            < config.minimum_box_height_ratio
            <= 1.0
        ):
            raise PlayerTrackingError(
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
            raise PlayerTrackingError(
                "Frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise PlayerTrackingError(
                "Frame must be a NumPy array."
            )

        if frame.size == 0:
            raise PlayerTrackingError(
                "Frame cannot be empty."
            )

        if frame.ndim != 3:
            raise PlayerTrackingError(
                "Frame must contain height, width, and channels."
            )

        if frame.shape[2] != 3:
            raise PlayerTrackingError(
                "Frame must contain exactly 3 channels."
            )

    # ============================================================
    # BYTE TRACKING
    # ============================================================

    def track_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
    ) -> List[Dict[str, Any]]:

        self.validate_frame(frame)

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.config.tracker_config,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=[
                    self.config.person_class_id
                ],
                device=self.config.device,
                verbose=False,
            )

        except Exception as error:
            raise PlayerTrackingError(
                f"Player tracking inference failed: {error}"
            ) from error

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes

        if len(boxes) == 0:
            return []

        if boxes.id is None:
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

        track_ids = (
            boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        tracks: List[
            Dict[str, Any]
        ] = []

        for index in range(
            len(xyxy)
        ):

            x1, y1, x2, y2 = [
                float(value)
                for value in xyxy[index]
            ]

            track_id = int(
                track_ids[index]
            )

            confidence = float(
                confidences[index]
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

            tracks.append(
                {
                    "track_id": track_id,

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
                            center_x
                            / frame_width,
                            6,
                        ),

                        "center_y": round(
                            center_y
                            / frame_height,
                            6,
                        ),
                    },

                    "frame_number": frame_number,

                    "timestamp_seconds": round(
                        timestamp_seconds,
                        3,
                    ),
                }
            )

        return tracks

    # ============================================================
    # COURT FILTERING
    # ============================================================

    def filter_court_tracks(
        self,
        tracks: List[
            Dict[str, Any]
        ],
        frame_height: int,
    ) -> List[Dict[str, Any]]:

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

        for track in tracks:

            foot_y = float(
                track[
                    "foot_point"
                ]["y"]
            )

            box_height = float(
                track[
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
                track
            )

        return filtered

    # ============================================================
    # PLAYER CANDIDATE SCORE
    # ============================================================

    @staticmethod
    def _selection_score(
        track: Dict[str, Any],
    ) -> float:

        confidence = float(
            track["confidence"]
        )

        box_width = float(
            track[
                "bbox"
            ]["width"]
        )

        box_height = float(
            track[
                "bbox"
            ]["height"]
        )

        area = max(
            box_width * box_height,
            1.0,
        )

        score = (
            confidence
            * np.sqrt(area)
        )

        return float(score)

    # ============================================================
    # PLAYER SELECTION
    # ============================================================

    def select_player_tracks(
        self,
        tracks: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:

        candidates: List[
            Dict[str, Any]
        ] = []

        for track in tracks:

            candidate = dict(
                track
            )

            candidate[
                "selection_score"
            ] = round(
                self._selection_score(
                    track
                ),
                4,
            )

            candidates.append(
                candidate
            )

        candidates.sort(
            key=lambda item: item[
                "selection_score"
            ],
            reverse=True,
        )

        return candidates[
            : self.config.max_players
        ]

    # ============================================================
    # PLAYER A / PLAYER B ASSIGNMENT
    # ============================================================

    def assign_stable_labels(
        self,
        tracks: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:

        if not tracks:
            return []

        # Far player normally appears higher in frame.
        ordered = sorted(
            tracks,
            key=lambda item: item[
                "foot_point"
            ]["y"],
        )

        labelled: List[
            Dict[str, Any]
        ] = []
        
        # Track which labels are used in this frame (not historical)
        labels_used_this_frame = set()

        for position, track in enumerate(
            ordered
        ):

            track_id = int(
                track["track_id"]
            )

            item = dict(
                track
            )

            # If this track_id has been seen before, reuse its label
            if (
                track_id
                in self.track_to_player_label
            ):
                label = (
                    self.track_to_player_label[
                        track_id
                    ]
                )
                
                # CRITICAL FIX: If this label was already used by another track
                # in this frame, we need to reassign
                if label in labels_used_this_frame and label in ["Player A", "Player B"]:
                    # This track_id had a label, but another active track is using it
                    # Assign the other player label
                    if label == "Player A" and "Player B" not in labels_used_this_frame:
                        label = "Player B"
                        self.track_to_player_label[track_id] = label
                    elif label == "Player B" and "Player A" not in labels_used_this_frame:
                        label = "Player A"
                        self.track_to_player_label[track_id] = label
                    else:
                        # Both taken, use track ID
                        label = f"Track {track_id}"
                        self.track_to_player_label[track_id] = label

            else:
                # New track_id - assign a label
                # Check which labels are available (not used by current active tracks)
                
                if (
                    "Player A"
                    not in labels_used_this_frame
                    and position == 0
                ):
                    label = "Player A"

                elif (
                    "Player B"
                    not in labels_used_this_frame
                ):
                    label = "Player B"

                elif (
                    "Player A"
                    not in labels_used_this_frame
                ):
                    label = "Player A"

                else:
                    # Both labels taken by current frame tracks
                    label = (
                        f"Track {track_id}"
                    )

                self.track_to_player_label[
                    track_id
                ] = label
            
            # Mark this label as used in current frame
            labels_used_this_frame.add(label)

            item[
                "player_label"
            ] = label

            if label == "Player A":
                item[
                    "court_side"
                ] = "far"

            elif label == "Player B":
                item[
                    "court_side"
                ] = "near"

            else:
                item[
                    "court_side"
                ] = "unknown"

            labelled.append(
                item
            )

        return labelled

    # ============================================================
    # HISTORY
    # ============================================================

    def update_history(
        self,
        players: List[
            Dict[str, Any]
        ],
    ) -> None:

        for player in players:

            label = player.get(
                "player_label"
            )

            if (
                label
                not in self.player_history
            ):
                continue

            self.player_history[
                label
            ].append(
                {
                    "track_id": player[
                        "track_id"
                    ],

                    "frame_number": player[
                        "frame_number"
                    ],

                    "timestamp_seconds": player[
                        "timestamp_seconds"
                    ],

                    "center": player[
                        "center"
                    ],

                    "foot_point": player[
                        "foot_point"
                    ],

                    "confidence": player[
                        "confidence"
                    ],
                }
            )

    # ============================================================
    # COMPLETE FRAME PROCESSING
    # ============================================================

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
    ) -> Dict[str, Any]:

        raw_tracks = self.track_frame(
            frame,
            frame_number,
            timestamp_seconds,
        )

        court_tracks = (
            self.filter_court_tracks(
                raw_tracks,
                frame.shape[0],
            )
        )

        selected_tracks = (
            self.select_player_tracks(
                court_tracks
            )
        )

        players = (
            self.assign_stable_labels(
                selected_tracks
            )
        )

        self.update_history(
            players
        )

        return {
            "raw_track_count": len(
                raw_tracks
            ),

            "court_track_count": len(
                court_tracks
            ),

            "player_count": len(
                players
            ),

            "players": players,
        }

    # ============================================================
    # DRAW PLAYER BOXES
    # ============================================================

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        players: List[
            Dict[str, Any]
        ],
    ) -> np.ndarray:

        PlayerTracker.validate_frame(
            frame
        )

        overlay = frame.copy()

        for player in players:

            bbox = player[
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

            label = player.get(
                "player_label",
                "Player",
            )

            track_id = player.get(
                "track_id",
                -1,
            )

            confidence = float(
                player[
                    "confidence"
                ]
            )

            text = (
                f"{label} | "
                f"ID {track_id} | "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                overlay,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

            cv2.putText(
                overlay,
                text,
                (
                    x1,
                    max(
                        y1 - 10,
                        25,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    0,
                    255,
                    0,
                ),
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
                (
                    0,
                    0,
                    255,
                ),
                -1,
            )

        return overlay

    # ============================================================
    # DRAW MOVEMENT HISTORY
    # ============================================================

    def draw_history(
        self,
        frame: np.ndarray,
        max_points: int = 30,
    ) -> np.ndarray:

        self.validate_frame(
            frame
        )

        overlay = frame.copy()

        for (
            player_label,
            history,
        ) in self.player_history.items():

            if len(history) < 2:
                continue

            recent_history = history[
                -max_points:
            ]

            points = []

            for item in recent_history:

                foot = item[
                    "foot_point"
                ]

                points.append(
                    (
                        int(
                            round(
                                foot["x"]
                            )
                        ),

                        int(
                            round(
                                foot["y"]
                            )
                        ),
                    )
                )

            for index in range(
                1,
                len(points),
            ):

                cv2.line(
                    overlay,
                    points[
                        index - 1
                    ],
                    points[index],
                    (
                        255,
                        255,
                        0,
                    ),
                    2,
                    cv2.LINE_AA,
                )

        return overlay

    # ============================================================
    # STATE RESET
    # ============================================================

    def reset(self) -> None:

        self.track_to_player_label.clear()

        self.player_history = {
            "Player A": [],
            "Player B": [],
        }