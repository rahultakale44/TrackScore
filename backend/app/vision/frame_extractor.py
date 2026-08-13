from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .video_loader import VideoLoader, VideoLoaderError


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""


class FrameExtractor:
    """
    Extracts frames from a video using frame intervals or timestamps.
    """

    def __init__(self, video_path: str, output_dir: str):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.loader = VideoLoader(video_path)

    def _prepare_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def frame_to_timestamp(frame_number: int, fps: float) -> float:
        if fps <= 0:
            raise FrameExtractionError("FPS must be greater than zero.")

        if frame_number < 0:
            raise FrameExtractionError(
                "Frame number cannot be negative."
            )

        return frame_number / fps

    @staticmethod
    def timestamp_to_frame(timestamp_seconds: float, fps: float) -> int:
        if fps <= 0:
            raise FrameExtractionError("FPS must be greater than zero.")

        if timestamp_seconds < 0:
            raise FrameExtractionError(
                "Timestamp cannot be negative."
            )

        return int(round(timestamp_seconds * fps))

    @staticmethod
    def format_timestamp(timestamp_seconds: float) -> str:
        total_milliseconds = int(round(timestamp_seconds * 1000))

        hours = total_milliseconds // 3_600_000
        remaining = total_milliseconds % 3_600_000

        minutes = remaining // 60_000
        remaining %= 60_000

        seconds = remaining // 1000
        milliseconds = remaining % 1000

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}."
            f"{milliseconds:03d}"
        )

    def _save_frame(
        self,
        frame,
        frame_number: int,
        timestamp_seconds: float,
        prefix: str = "frame",
    ) -> Dict[str, Any]:
        self._prepare_output_directory()

        filename = (
            f"{prefix}_"
            f"{frame_number:06d}_"
            f"{timestamp_seconds:.3f}s.jpg"
        )

        output_path = self.output_dir / filename

        success = cv2.imwrite(str(output_path), frame)

        if not success:
            raise FrameExtractionError(
                f"Failed to save frame: {output_path}"
            )

        return {
            "frame_number": frame_number,
            "timestamp_seconds": round(timestamp_seconds, 3),
            "timestamp_formatted": self.format_timestamp(
                timestamp_seconds
            ),
            "filename": filename,
            "path": str(output_path.resolve()),
        }

    def extract_every_n_frames(
        self,
        interval: int,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if interval <= 0:
            raise FrameExtractionError(
                "Frame interval must be greater than zero."
            )

        if max_frames is not None and max_frames <= 0:
            raise FrameExtractionError(
                "max_frames must be greater than zero."
            )

        capture = self.loader.open_video()

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))

            if fps <= 0:
                raise FrameExtractionError(
                    "Unable to determine valid video FPS."
                )

            extracted_frames: List[Dict[str, Any]] = []

            frame_number = 0
            saved_count = 0

            while True:
                success, frame = capture.read()

                if not success or frame is None:
                    break

                if frame_number % interval == 0:
                    timestamp_seconds = self.frame_to_timestamp(
                        frame_number,
                        fps,
                    )

                    metadata = self._save_frame(
                        frame=frame,
                        frame_number=frame_number,
                        timestamp_seconds=timestamp_seconds,
                    )

                    extracted_frames.append(metadata)
                    saved_count += 1

                    if (
                        max_frames is not None
                        and saved_count >= max_frames
                    ):
                        break

                frame_number += 1

            return extracted_frames

        finally:
            capture.release()

    def extract_at_timestamps(
        self,
        timestamps: List[float],
    ) -> List[Dict[str, Any]]:
        if not timestamps:
            raise FrameExtractionError(
                "At least one timestamp is required."
            )

        metadata = self.loader.get_metadata()
        duration_seconds = float(metadata["duration_seconds"])
        fps = float(metadata["fps"])

        capture = self.loader.open_video()

        extracted_frames: List[Dict[str, Any]] = []

        try:
            for timestamp_seconds in timestamps:
                if timestamp_seconds < 0:
                    raise FrameExtractionError(
                        f"Invalid negative timestamp: "
                        f"{timestamp_seconds}"
                    )

                if timestamp_seconds > duration_seconds:
                    raise FrameExtractionError(
                        f"Timestamp {timestamp_seconds}s exceeds "
                        f"video duration of {duration_seconds}s."
                    )

                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp_seconds * 1000,
                )

                success, frame = capture.read()

                if not success or frame is None:
                    raise FrameExtractionError(
                        f"Could not read frame at "
                        f"{timestamp_seconds}s."
                    )

                actual_frame_number = int(
                    capture.get(cv2.CAP_PROP_POS_FRAMES)
                ) - 1

                if actual_frame_number < 0:
                    actual_frame_number = self.timestamp_to_frame(
                        timestamp_seconds,
                        fps,
                    )

                actual_timestamp = self.frame_to_timestamp(
                    actual_frame_number,
                    fps,
                )

                frame_metadata = self._save_frame(
                    frame=frame,
                    frame_number=actual_frame_number,
                    timestamp_seconds=actual_timestamp,
                    prefix="timestamp",
                )

                extracted_frames.append(frame_metadata)

            return extracted_frames

        finally:
            capture.release()

    def save_metadata_json(
        self,
        extracted_frames: List[Dict[str, Any]],
        filename: str = "frames_metadata.json",
    ) -> Path:
        self._prepare_output_directory()

        metadata_path = self.output_dir / filename

        payload = {
            "video": str(self.video_path.resolve()),
            "total_extracted_frames": len(extracted_frames),
            "frames": extracted_frames,
        }

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=4,
            )

        return metadata_path