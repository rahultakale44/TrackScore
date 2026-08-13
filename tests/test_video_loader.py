from pathlib import Path

import pytest

from backend.app.vision.video_loader import (
    SUPPORTED_VIDEO_EXTENSIONS,
    VideoLoader,
    VideoLoaderError,
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