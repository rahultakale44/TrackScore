from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


class FramePreprocessingError(Exception):
    """Raised when frame preprocessing fails."""


@dataclass
class FrameQualityConfig:
    """
    Threshold configuration for basic frame-quality analysis.
    """

    dark_threshold: float = 45.0
    bright_threshold: float = 210.0
    blur_threshold: float = 80.0


class FramePreprocessor:
    """
    Prepares tennis video frames for machine-learning inference.

    Responsibilities:
    - Validate image frames
    - Resize while preserving aspect ratio
    - Apply letterboxing
    - Measure frame brightness
    - Measure frame sharpness
    - Identify blurry/dark/bright frames
    - Convert BGR to RGB
    - Normalize image values to [0, 1]
    - Convert model input to CHW format
    """

    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 640,
        quality_config: FrameQualityConfig | None = None,
    ):
        if target_width <= 0 or target_height <= 0:
            raise FramePreprocessingError(
                "Target width and height must be greater than zero."
            )

        self.target_width = target_width
        self.target_height = target_height

        self.quality_config = (
            quality_config
            if quality_config is not None
            else FrameQualityConfig()
        )

    @staticmethod
    def validate_frame(frame: np.ndarray) -> None:
        """
        Validate an OpenCV image frame.
        """
        if frame is None:
            raise FramePreprocessingError(
                "Frame cannot be None."
            )

        if not isinstance(frame, np.ndarray):
            raise FramePreprocessingError(
                "Frame must be a NumPy array."
            )

        if frame.size == 0:
            raise FramePreprocessingError(
                "Frame cannot be empty."
            )

        if frame.ndim != 3:
            raise FramePreprocessingError(
                "Frame must contain height, width, and channels."
            )

        if frame.shape[2] != 3:
            raise FramePreprocessingError(
                "Frame must contain exactly 3 color channels."
            )

    def resize_with_letterbox(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resize a frame while preserving aspect ratio.

        Remaining space is filled using letterbox padding.
        """
        self.validate_frame(frame)

        original_height, original_width = frame.shape[:2]

        scale = min(
            self.target_width / original_width,
            self.target_height / original_height,
        )

        resized_width = max(
            1,
            int(round(original_width * scale)),
        )

        resized_height = max(
            1,
            int(round(original_height * scale)),
        )

        resized_frame = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        horizontal_padding = (
            self.target_width - resized_width
        )

        vertical_padding = (
            self.target_height - resized_height
        )

        pad_left = horizontal_padding // 2
        pad_right = horizontal_padding - pad_left

        pad_top = vertical_padding // 2
        pad_bottom = vertical_padding - pad_top

        letterboxed_frame = cv2.copyMakeBorder(
            resized_frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        metadata = {
            "original_width": original_width,
            "original_height": original_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "scale_factor": round(scale, 6),
            "padding": {
                "left": pad_left,
                "right": pad_right,
                "top": pad_top,
                "bottom": pad_bottom,
            },
        }

        return letterboxed_frame, metadata

    @staticmethod
    def calculate_brightness(
        frame: np.ndarray,
    ) -> float:
        """
        Calculate average grayscale brightness.
        """
        FramePreprocessor.validate_frame(frame)

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(np.mean(gray_frame))

        return round(brightness, 2)

    @staticmethod
    def calculate_sharpness(
        frame: np.ndarray,
    ) -> float:
        """
        Calculate sharpness using Laplacian variance.

        Higher value generally indicates a sharper frame.
        """
        FramePreprocessor.validate_frame(frame)

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        laplacian = cv2.Laplacian(
            gray_frame,
            cv2.CV_64F,
        )

        sharpness = float(laplacian.var())

        return round(sharpness, 2)

    def assess_quality(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyse basic visual quality of a frame.
        """
        brightness = self.calculate_brightness(frame)
        sharpness = self.calculate_sharpness(frame)

        is_dark = (
            brightness
            < self.quality_config.dark_threshold
        )

        is_bright = (
            brightness
            > self.quality_config.bright_threshold
        )

        is_blurry = (
            sharpness
            < self.quality_config.blur_threshold
        )

        issues = []

        if is_dark:
            issues.append("dark")

        if is_bright:
            issues.append("overexposed")

        if is_blurry:
            issues.append("blurry")

        acceptable = len(issues) == 0

        return {
            "brightness": brightness,
            "sharpness": sharpness,
            "is_dark": is_dark,
            "is_overexposed": is_bright,
            "is_blurry": is_blurry,
            "acceptable": acceptable,
            "issues": issues,
        }

    @staticmethod
    def normalize_frame(
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convert uint8 image values from 0-255 to float32 0-1.
        """
        FramePreprocessor.validate_frame(frame)

        normalized = frame.astype(
            np.float32
        ) / 255.0

        return normalized

    @staticmethod
    def convert_bgr_to_rgb(
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convert OpenCV BGR image into RGB format.
        """
        FramePreprocessor.validate_frame(frame)

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

    @staticmethod
    def convert_hwc_to_chw(
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convert:
        Height x Width x Channels

        into:

        Channels x Height x Width
        """
        FramePreprocessor.validate_frame(frame)

        return np.transpose(
            frame,
            (2, 0, 1),
        )

    def prepare_model_input(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Produce model-ready input.

        Output shape:
        (1, 3, target_height, target_width)

        Values:
        float32 within [0, 1]
        """
        processed_frame, _ = (
            self.resize_with_letterbox(frame)
        )

        rgb_frame = self.convert_bgr_to_rgb(
            processed_frame
        )

        normalized_frame = self.normalize_frame(
            rgb_frame
        )

        chw_frame = np.transpose(
            normalized_frame,
            (2, 0, 1),
        )

        batch_frame = np.expand_dims(
            chw_frame,
            axis=0,
        )

        return np.ascontiguousarray(
            batch_frame,
            dtype=np.float32,
        )

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run complete preprocessing and quality analysis.
        """
        self.validate_frame(frame)

        quality = self.assess_quality(frame)

        processed_frame, resize_metadata = (
            self.resize_with_letterbox(frame)
        )

        model_input = self.prepare_model_input(
            frame
        )

        metadata = {
            "quality": quality,
            "resize": resize_metadata,
            "model_input": {
                "shape": list(model_input.shape),
                "dtype": str(model_input.dtype),
                "minimum_value": round(
                    float(model_input.min()),
                    4,
                ),
                "maximum_value": round(
                    float(model_input.max()),
                    4,
                ),
            },
        }

        return processed_frame, metadata

    @staticmethod
    def save_processed_frame(
        frame: np.ndarray,
        output_path: str,
    ) -> Path:
        """
        Save a processed preview frame.
        """
        FramePreprocessor.validate_frame(frame)

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output),
            frame,
        )

        if not success:
            raise FramePreprocessingError(
                f"Unable to save processed frame: {output}"
            )

        return output