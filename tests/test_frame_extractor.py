import pytest

from backend.app.vision.frame_extractor import (
    FrameExtractionError,
    FrameExtractor,
)


def test_frame_to_timestamp():
    assert FrameExtractor.frame_to_timestamp(
        frame_number=30,
        fps=30.0,
    ) == 1.0

    assert FrameExtractor.frame_to_timestamp(
        frame_number=75,
        fps=25.0,
    ) == 3.0


def test_timestamp_to_frame():
    assert FrameExtractor.timestamp_to_frame(
        timestamp_seconds=2.0,
        fps=30.0,
    ) == 60

    assert FrameExtractor.timestamp_to_frame(
        timestamp_seconds=1.5,
        fps=24.0,
    ) == 36


def test_format_timestamp():
    assert (
        FrameExtractor.format_timestamp(65.125)
        == "00:01:05.125"
    )

    assert (
        FrameExtractor.format_timestamp(3661.999)
        == "01:01:01.999"
    )


def test_negative_frame_number():
    with pytest.raises(FrameExtractionError):
        FrameExtractor.frame_to_timestamp(
            frame_number=-1,
            fps=30.0,
        )


def test_invalid_fps_frame_conversion():
    with pytest.raises(FrameExtractionError):
        FrameExtractor.frame_to_timestamp(
            frame_number=30,
            fps=0,
        )


def test_negative_timestamp():
    with pytest.raises(FrameExtractionError):
        FrameExtractor.timestamp_to_frame(
            timestamp_seconds=-1,
            fps=30.0,
        )


def test_invalid_fps_timestamp_conversion():
    with pytest.raises(FrameExtractionError):
        FrameExtractor.timestamp_to_frame(
            timestamp_seconds=1,
            fps=0,
        )