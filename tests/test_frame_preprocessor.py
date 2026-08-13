import numpy as np
import pytest

from backend.app.vision.frame_preprocessor import (
    FramePreprocessingError,
    FramePreprocessor,
)


@pytest.fixture
def sample_frame():
    return np.full(
        (720, 1280, 3),
        128,
        dtype=np.uint8,
    )


def test_validate_valid_frame(sample_frame):
    FramePreprocessor.validate_frame(
        sample_frame
    )


def test_validate_none_frame():
    with pytest.raises(
        FramePreprocessingError
    ):
        FramePreprocessor.validate_frame(
            None
        )


def test_validate_empty_frame():
    empty_frame = np.array([])

    with pytest.raises(
        FramePreprocessingError
    ):
        FramePreprocessor.validate_frame(
            empty_frame
        )


def test_resize_with_letterbox(
    sample_frame,
):
    preprocessor = FramePreprocessor(
        target_width=640,
        target_height=640,
    )

    processed, metadata = (
        preprocessor.resize_with_letterbox(
            sample_frame
        )
    )

    assert processed.shape == (
        640,
        640,
        3,
    )

    assert (
        metadata["target_width"]
        == 640
    )

    assert (
        metadata["target_height"]
        == 640
    )


def test_normalization(sample_frame):
    normalized = (
        FramePreprocessor.normalize_frame(
            sample_frame
        )
    )

    assert normalized.dtype == np.float32

    assert normalized.min() >= 0.0

    assert normalized.max() <= 1.0


def test_model_input_shape(
    sample_frame,
):
    preprocessor = FramePreprocessor(
        target_width=640,
        target_height=640,
    )

    model_input = (
        preprocessor.prepare_model_input(
            sample_frame
        )
    )

    assert model_input.shape == (
        1,
        3,
        640,
        640,
    )

    assert (
        model_input.dtype
        == np.float32
    )


def test_brightness_calculation():
    bright_frame = np.full(
        (100, 100, 3),
        200,
        dtype=np.uint8,
    )

    brightness = (
        FramePreprocessor.calculate_brightness(
            bright_frame
        )
    )

    assert brightness == 200.0


def test_dark_frame_detection():
    dark_frame = np.full(
        (100, 100, 3),
        10,
        dtype=np.uint8,
    )

    preprocessor = FramePreprocessor()

    quality = (
        preprocessor.assess_quality(
            dark_frame
        )
    )

    assert quality["is_dark"] is True


def test_overexposed_frame_detection():
    bright_frame = np.full(
        (100, 100, 3),
        250,
        dtype=np.uint8,
    )

    preprocessor = FramePreprocessor()

    quality = (
        preprocessor.assess_quality(
            bright_frame
        )
    )

    assert (
        quality["is_overexposed"]
        is True
    )


def test_invalid_target_dimensions():
    with pytest.raises(
        FramePreprocessingError
    ):
        FramePreprocessor(
            target_width=0,
            target_height=640,
        )