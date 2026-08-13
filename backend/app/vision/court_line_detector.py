from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


class CourtLineDetectionError(Exception):
    """Raised when tennis court line detection fails."""


@dataclass
class CourtLineConfig:
    """
    Configuration for tennis court line enhancement
    and candidate line detection.
    """

    # Bright/white line detection
    white_threshold: int = 170
    max_white_saturation: int = 95

    # Edge detection
    canny_low: int = 50
    canny_high: int = 150

    # Noise reduction
    blur_kernel_size: int = 5

    # Morphological cleanup
    morph_kernel_width: int = 3
    morph_kernel_height: int = 3

    # Hough line detection
    hough_threshold: int = 55
    min_line_length: int = 70
    max_line_gap: int = 20

    # Orientation classification
    horizontal_tolerance: float = 15.0
    vertical_tolerance: float = 15.0

    # Region of interest
    roi_top_ratio: float = 0.20
    roi_bottom_ratio: float = 1.00


class CourtLineDetector:
    """
    Detects candidate tennis-court lines from a video frame.

    Pipeline:
        Frame
          ↓
        ROI Mask
          ↓
        White-Line Enhancement
          ↓
        Morphological Cleanup
          ↓
        Gaussian Blur
          ↓
        Canny Edge Detection
          ↓
        Probabilistic Hough Transform
          ↓
        Candidate Court Lines
          ↓
        Orientation Classification

    Important:
    This module generates candidate geometry.

    Final court identification will later combine these
    candidates with court keypoints, known tennis geometry,
    and homography estimation.
    """

    def __init__(
        self,
        config: CourtLineConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else CourtLineConfig()
        )

        self._validate_config()

    # ============================================================
    # CONFIGURATION VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if not 0 <= config.white_threshold <= 255:
            raise CourtLineDetectionError(
                "white_threshold must be between 0 and 255."
            )

        if not 0 <= config.max_white_saturation <= 255:
            raise CourtLineDetectionError(
                "max_white_saturation must be between 0 and 255."
            )

        if config.canny_low < 0:
            raise CourtLineDetectionError(
                "canny_low cannot be negative."
            )

        if config.canny_high < 0:
            raise CourtLineDetectionError(
                "canny_high cannot be negative."
            )

        if config.canny_low >= config.canny_high:
            raise CourtLineDetectionError(
                "canny_low must be smaller than canny_high."
            )

        if config.blur_kernel_size <= 0:
            raise CourtLineDetectionError(
                "blur_kernel_size must be greater than zero."
            )

        if config.blur_kernel_size % 2 == 0:
            raise CourtLineDetectionError(
                "blur_kernel_size must be odd."
            )

        if config.morph_kernel_width <= 0:
            raise CourtLineDetectionError(
                "morph_kernel_width must be greater than zero."
            )

        if config.morph_kernel_height <= 0:
            raise CourtLineDetectionError(
                "morph_kernel_height must be greater than zero."
            )

        if config.hough_threshold <= 0:
            raise CourtLineDetectionError(
                "hough_threshold must be greater than zero."
            )

        if config.min_line_length <= 0:
            raise CourtLineDetectionError(
                "min_line_length must be greater than zero."
            )

        if config.max_line_gap < 0:
            raise CourtLineDetectionError(
                "max_line_gap cannot be negative."
            )

        if config.horizontal_tolerance < 0:
            raise CourtLineDetectionError(
                "horizontal_tolerance cannot be negative."
            )

        if config.vertical_tolerance < 0:
            raise CourtLineDetectionError(
                "vertical_tolerance cannot be negative."
            )

        if not 0.0 <= config.roi_top_ratio < 1.0:
            raise CourtLineDetectionError(
                "roi_top_ratio must be within [0, 1)."
            )

        if not 0.0 < config.roi_bottom_ratio <= 1.0:
            raise CourtLineDetectionError(
                "roi_bottom_ratio must be within (0, 1]."
            )

        if (
            config.roi_top_ratio
            >= config.roi_bottom_ratio
        ):
            raise CourtLineDetectionError(
                "roi_top_ratio must be smaller than roi_bottom_ratio."
            )

    # ============================================================
    # FRAME VALIDATION
    # ============================================================

    @staticmethod
    def validate_frame(
        frame: np.ndarray,
    ) -> None:
        if frame is None:
            raise CourtLineDetectionError(
                "Frame cannot be None."
            )

        if not isinstance(frame, np.ndarray):
            raise CourtLineDetectionError(
                "Frame must be a NumPy array."
            )

        if frame.size == 0:
            raise CourtLineDetectionError(
                "Frame cannot be empty."
            )

        if frame.ndim != 3:
            raise CourtLineDetectionError(
                "Frame must have height, width, and channels."
            )

        if frame.shape[2] != 3:
            raise CourtLineDetectionError(
                "Frame must contain exactly 3 color channels."
            )

    # ============================================================
    # ROI
    # ============================================================

    def create_roi_mask(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a broad region-of-interest mask.

        The upper broadcast area is partially ignored to reduce
        audience, scoreboard, roof, and background detections.
        """

        self.validate_frame(frame)

        height, width = frame.shape[:2]

        top_y = int(
            round(
                height
                * self.config.roi_top_ratio
            )
        )

        bottom_y = int(
            round(
                height
                * self.config.roi_bottom_ratio
            )
        )

        bottom_y = min(
            bottom_y,
            height,
        )

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        polygon = np.array(
            [
                [
                    (0, top_y),
                    (width - 1, top_y),
                    (width - 1, bottom_y - 1),
                    (0, bottom_y - 1),
                ]
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(
            mask,
            polygon,
            255,
        )

        return mask

    # ============================================================
    # WHITE COURT LINE ENHANCEMENT
    # ============================================================

    def enhance_white_lines(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Highlight bright low-saturation pixels likely to
        correspond to tennis-court markings.
        """

        self.validate_frame(frame)

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        lower_white = np.array(
            [
                0,
                0,
                self.config.white_threshold,
            ],
            dtype=np.uint8,
        )

        upper_white = np.array(
            [
                179,
                self.config.max_white_saturation,
                255,
            ],
            dtype=np.uint8,
        )

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white,
        )

        roi_mask = self.create_roi_mask(
            frame
        )

        white_mask = cv2.bitwise_and(
            white_mask,
            roi_mask,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                self.config.morph_kernel_width,
                self.config.morph_kernel_height,
            ),
        )

        cleaned_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        return cleaned_mask

    # ============================================================
    # EDGE DETECTION
    # ============================================================

    def create_edge_map(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Generate an edge map focused mainly on court markings.
        """

        line_mask = self.enhance_white_lines(
            frame
        )

        blurred = cv2.GaussianBlur(
            line_mask,
            (
                self.config.blur_kernel_size,
                self.config.blur_kernel_size,
            ),
            0,
        )

        edges = cv2.Canny(
            blurred,
            self.config.canny_low,
            self.config.canny_high,
        )

        return edges

    # ============================================================
    # LINE GEOMETRY
    # ============================================================

    @staticmethod
    def calculate_line_length(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> float:
        """
        Calculate Euclidean length of a line in pixels.
        """

        return float(
            hypot(
                x2 - x1,
                y2 - y1,
            )
        )

    @staticmethod
    def calculate_line_angle(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> float:
        """
        Calculate line orientation inside [0, 180) degrees.
        """

        angle = degrees(
            atan2(
                y2 - y1,
                x2 - x1,
            )
        )

        angle %= 180.0

        return angle

    def classify_orientation(
        self,
        angle: float,
    ) -> str:
        """
        Classify detected line orientation.

        horizontal:
            Approximately parallel to image x-axis.

        vertical:
            Approximately perpendicular to image x-axis.

        diagonal:
            Remaining perspective lines.
        """

        horizontal_distance = min(
            abs(angle),
            abs(180.0 - angle),
        )

        vertical_distance = abs(
            angle - 90.0
        )

        if (
            horizontal_distance
            <= self.config.horizontal_tolerance
        ):
            return "horizontal"

        if (
            vertical_distance
            <= self.config.vertical_tolerance
        ):
            return "vertical"

        return "diagonal"

    # ============================================================
    # HOUGH LINE DETECTION
    # ============================================================

    def detect_lines(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect candidate court line segments.

        Handles different OpenCV HoughLinesP return formats:

            (N, 1, 4)

        and

            (N, 4)

        without assuming a fixed nested structure.
        """

        self.validate_frame(frame)

        edges = self.create_edge_map(
            frame
        )

        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        detected_lines: List[
            Dict[str, Any]
        ] = []

        if raw_lines is None:
            return detected_lines

        for index, line in enumerate(
            raw_lines,
            start=1,
        ):
            coordinates = np.asarray(
                line
            ).reshape(-1)

            if coordinates.size != 4:
                continue

            x1, y1, x2, y2 = [
                int(value)
                for value in coordinates
            ]

            length = (
                self.calculate_line_length(
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            angle = (
                self.calculate_line_angle(
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            orientation = (
                self.classify_orientation(
                    angle
                )
            )

            midpoint_x = (
                x1 + x2
            ) / 2.0

            midpoint_y = (
                y1 + y2
            ) / 2.0

            detected_lines.append(
                {
                    "id": index,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "midpoint_x": round(
                        midpoint_x,
                        2,
                    ),
                    "midpoint_y": round(
                        midpoint_y,
                        2,
                    ),
                    "length_pixels": round(
                        length,
                        2,
                    ),
                    "angle_degrees": round(
                        angle,
                        2,
                    ),
                    "orientation": orientation,
                }
            )

        detected_lines.sort(
            key=lambda item: item[
                "length_pixels"
            ],
            reverse=True,
        )

        # Reassign IDs after sorting so they stay ordered.
        for index, line in enumerate(
            detected_lines,
            start=1,
        ):
            line["id"] = index

        return detected_lines

    # ============================================================
    # LINE SUMMARY
    # ============================================================

    def summarize_lines(
        self,
        lines: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce high-level statistics about detected lines.
        """

        counts = {
            "horizontal": 0,
            "vertical": 0,
            "diagonal": 0,
        }

        total_length = 0.0

        for line in lines:
            orientation = line.get(
                "orientation",
                "diagonal",
            )

            if orientation in counts:
                counts[orientation] += 1

            total_length += float(
                line.get(
                    "length_pixels",
                    0.0,
                )
            )

        longest_line = (
            lines[0]
            if lines
            else None
        )

        average_length = (
            total_length / len(lines)
            if lines
            else 0.0
        )

        return {
            "total_lines": len(lines),
            "horizontal_lines": counts[
                "horizontal"
            ],
            "vertical_lines": counts[
                "vertical"
            ],
            "diagonal_lines": counts[
                "diagonal"
            ],
            "average_line_length_pixels": round(
                average_length,
                2,
            ),
            "longest_line": longest_line,
        }

    # ============================================================
    # DEBUG OVERLAY
    # ============================================================

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw candidate court lines over the original frame.

        Color coding:

        Green  = Horizontal-like
        Blue   = Vertical-like
        Yellow = Diagonal
        """

        self.validate_frame(frame)

        overlay = frame.copy()

        for line in lines:
            orientation = line[
                "orientation"
            ]

            if orientation == "horizontal":
                color = (
                    0,
                    255,
                    0,
                )

            elif orientation == "vertical":
                color = (
                    255,
                    0,
                    0,
                )

            else:
                color = (
                    0,
                    255,
                    255,
                )

            cv2.line(
                overlay,
                (
                    line["x1"],
                    line["y1"],
                ),
                (
                    line["x2"],
                    line["y2"],
                ),
                color,
                2,
                cv2.LINE_AA,
            )

        summary = self.summarize_lines(
            lines
        )

        cv2.rectangle(
            overlay,
            (15, 15),
            (590, 145),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            overlay,
            "TrackScore - Court Geometry",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            (
                f"Candidates: "
                f"{summary['total_lines']}"
            ),
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            (
                "Horizontal: "
                f"{summary['horizontal_lines']}  "
                "Vertical: "
                f"{summary['vertical_lines']}"
            ),
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            (
                "Diagonal: "
                f"{summary['diagonal_lines']}"
            ),
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

    # ============================================================
    # COMPLETE ANALYSIS
    # ============================================================

    def analyse_frame(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run complete court-line analysis on one frame.
        """

        self.validate_frame(frame)

        line_mask = self.enhance_white_lines(
            frame
        )

        edge_map = self.create_edge_map(
            frame
        )

        lines = self.detect_lines(
            frame
        )

        summary = self.summarize_lines(
            lines
        )

        overlay = self.draw_debug_overlay(
            frame,
            lines,
        )

        return {
            "line_mask": line_mask,
            "edge_map": edge_map,
            "lines": lines,
            "summary": summary,
            "overlay": overlay,
        }

    # ============================================================
    # IMAGE SAVING
    # ============================================================

    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: str,
    ) -> Path:
        """
        Save an OpenCV image safely.
        """

        if image is None:
            raise CourtLineDetectionError(
                "Cannot save a None image."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise CourtLineDetectionError(
                "Image must be a NumPy array."
            )

        if image.size == 0:
            raise CourtLineDetectionError(
                "Cannot save an empty image."
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
            raise CourtLineDetectionError(
                f"Unable to save image: {output}"
            )

        return output