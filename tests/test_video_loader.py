from pathlib import Path

import numpy as np
import pytest

from backend.app.vision.video_loader import (
    SUPPORTED_VIDEO_EXTENSIONS,
    VideoLoader,
    VideoLoaderError,
)
from backend.app.vision.video_pipeline import (
    VideoAnalyticsPipeline,
)


def test_supported_video_extensions():
    expected_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    }

    assert SUPPORTED_VIDEO_EXTENSIONS == expected_extensions


def test_invalid_video_path():
    loader = VideoLoader("video_that_does_not_exist.mp4")

    with pytest.raises(VideoLoaderError):
        loader.validate_file()


def test_unsupported_extension(tmp_path: Path):
    fake_file = tmp_path / "sample.txt"
    fake_file.write_text("not a video")

    loader = VideoLoader(str(fake_file))

    with pytest.raises(VideoLoaderError):
        loader.validate_file()


def test_aspect_ratio():
    assert VideoLoader._calculate_aspect_ratio(1920, 1080) == "16:9"
    assert VideoLoader._calculate_aspect_ratio(1280, 720) == "16:9"
    assert VideoLoader._calculate_aspect_ratio(640, 480) == "4:3"


def test_duration_formatting():
    assert VideoLoader._format_duration(65) == "00:01:05"
    assert VideoLoader._format_duration(3661) == "01:01:01"


def test_unified_video_analytics_pipeline(tmp_path):
    class FakeVideoLoader:
        def __init__(self, video_path):
            self.video_path = video_path

        def get_metadata(self):
            return {
                "filename": "demo.mp4",
                "fps": 30.0,
                "frame_count": 90,
                "duration_seconds": 3.0,
                "resolution": "1280x720",
            }

    class FakeFrameExtractor:
        def __init__(self, video_path, output_dir):
            self.video_path = video_path
            self.output_dir = Path(output_dir)

        def extract_every_n_frames(self, interval, max_frames=None):
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            return [{
                "frame_number": 0,
                "timestamp_seconds": 0.0,
                "filename": "sample_0.jpg",
                "frame": frame,
                "path": str(self.output_dir / "sample_0.jpg"),
            }]

    class FakeFramePreprocessor:
        def __init__(self, target_width=640, target_height=640, quality_config=None):
            self.target_width = target_width
            self.target_height = target_height

        def process_frame(self, frame):
            return frame, {
                "quality": {"acceptable": True},
                "resize": {"target_width": self.target_width},
            }

    class FakePlayerDetector:
        def __init__(self, config=None):
            self.config = config

        def detect_players(self, frame):
            return {"players": [{"id": "p1", "bbox": [1, 2, 3, 4]}]}

    class FakeBallDetector:
        def __init__(self, config=None):
            self.config = config

        def detect_ball(self, frame):
            return {"balls": [{"x": 5, "y": 6}]}

    pipeline = VideoAnalyticsPipeline(
        video_path="demo.mp4",
        output_dir=str(tmp_path),
        video_loader_cls=FakeVideoLoader,
        frame_extractor_cls=FakeFrameExtractor,
        preprocessor_cls=FakeFramePreprocessor,
        player_detector_cls=FakePlayerDetector,
        ball_detector_cls=FakeBallDetector,
    )

    result = pipeline.run(max_frames=1)

    assert result["video_metadata"]["filename"] == "demo.mp4"
    assert result["frame_count"] == 1
    assert result["detections"]["players"][0]["id"] == "p1"
    assert result["detections"]["ball"][0]["x"] == 5
    assert result["status"] == "success"