from pathlib import Path
from typing import Dict, Any

import cv2


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VideoLoaderError(Exception):
    """Custom exception for video loading and validation errors."""


class VideoLoader:
    """
    Handles video validation, loading, and metadata extraction.

    This class is intentionally focused only on video ingestion.
    Detection, tracking, ML inference, and analytics will be handled
    by separate modules later.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

    def validate_file(self) -> None:
        """
        Validate that the supplied file exists and has a supported extension.
        """
        if not self.video_path.exists():
            raise VideoLoaderError(
                f"Video file does not exist: {self.video_path}"
            )

        if not self.video_path.is_file():
            raise VideoLoaderError(
                f"Provided path is not a file: {self.video_path}"
            )

        extension = self.video_path.suffix.lower()

        if extension not in SUPPORTED_VIDEO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
            raise VideoLoaderError(
                f"Unsupported video format '{extension}'. "
                f"Supported formats: {supported}"
            )

    def open_video(self) -> cv2.VideoCapture:
        """
        Open the video using OpenCV.
        """
        self.validate_file()

        capture = cv2.VideoCapture(str(self.video_path))

        if not capture.isOpened():
            capture.release()
            raise VideoLoaderError(
                f"Unable to open video: {self.video_path}"
            )

        return capture

    @staticmethod
    def _decode_fourcc(fourcc_value: int) -> str:
        """
        Convert OpenCV FOURCC integer into readable codec text.
        """
        if fourcc_value <= 0:
            return "Unknown"

        codec = "".join(
            chr((fourcc_value >> (8 * i)) & 0xFF)
            for i in range(4)
        )

        codec = codec.strip()

        return codec if codec else "Unknown"

    @staticmethod
    def _calculate_aspect_ratio(width: int, height: int) -> str:
        """
        Return a simplified aspect ratio such as 16:9 or 4:3.
        """
        if width <= 0 or height <= 0:
            return "Unknown"

        from math import gcd

        divisor = gcd(width, height)

        return f"{width // divisor}:{height // divisor}"

    @staticmethod
    def _format_duration(duration_seconds: float) -> str:
        """
        Convert seconds into HH:MM:SS format.
        """
        total_seconds = max(0, int(round(duration_seconds)))

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract important metadata from the video.
        """
        capture = self.open_video()

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))

            if fps <= 0:
                raise VideoLoaderError(
                    "Unable to determine valid FPS from the video."
                )

            if frame_count <= 0:
                raise VideoLoaderError(
                    "Unable to determine frame count from the video."
                )

            if width <= 0 or height <= 0:
                raise VideoLoaderError(
                    "Unable to determine valid video resolution."
                )

            duration_seconds = frame_count / fps
            file_size_bytes = self.video_path.stat().st_size

            metadata = {
                "filename": self.video_path.name,
                "absolute_path": str(self.video_path.resolve()),
                "extension": self.video_path.suffix.lower(),
                "file_size_bytes": file_size_bytes,
                "file_size_mb": round(
                    file_size_bytes / (1024 * 1024),
                    2,
                ),
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "aspect_ratio": self._calculate_aspect_ratio(
                    width,
                    height,
                ),
                "duration_seconds": round(duration_seconds, 2),
                "duration_formatted": self._format_duration(
                    duration_seconds
                ),
                "codec": self._decode_fourcc(fourcc_value),
            }

            return metadata

        finally:
            capture.release()

    def read_first_frame(self):
        """
        Read and return the first frame.

        This will be useful later for court detection,
        player detection, and preview generation.
        """
        capture = self.open_video()

        try:
            success, frame = capture.read()

            if not success or frame is None:
                raise VideoLoaderError(
                    "Unable to read the first frame from the video."
                )

            return frame

        finally:
            capture.release()