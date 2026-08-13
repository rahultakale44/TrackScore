import numpy as np
import pytest

from backend.app.vision.court_line_detector import (
    CourtLineConfig,
    CourtLineDetectionError,
    CourtLineDetector,
)


@pytest.fixture
def sample_frame():
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    return frame


def test_detector_creation():
    detector = CourtLineDetector()

    assert detector.config is not None


def test_invalid_white_threshold():
    config = CourtLineConfig(
        white_threshold=300
    )

    with pytest.raises(
        CourtLineDetectionError
    ):
        CourtLineDetector(config)


def test_invalid_canny_thresholds():
    config = CourtLineConfig(
        canny_low=200,
        canny_high=100,
    )

    with pytest.raises(
        CourtLineDetectionError
    ):
        CourtLineDetector(config)


def test_invalid_blur_kernel():
    config = CourtLineConfig(
        blur_kernel_size=4
    )

    with pytest.raises(
        CourtLineDetectionError
    ):
        CourtLineDetector(config)


def test_validate_none_frame():
    with pytest.raises(
        CourtLineDetectionError
    ):
        CourtLineDetector.validate_frame(
            None
        )


def test_roi_mask_shape(
    sample_frame,
):
    detector = CourtLineDetector()

    mask = detector.create_roi_mask(
        sample_frame
    )

    assert mask.shape == (
        480,
        640,
    )

    assert mask.dtype == np.uint8


def test_line_length():
    length = (
        CourtLineDetector
        .calculate_line_length(
            0,
            0,
            3,
            4,
        )
    )

    assert length == 5.0


def test_horizontal_angle():
    angle = (
        CourtLineDetector
        .calculate_line_angle(
            0,
            10,
            100,
            10,
        )
    )

    assert angle == 0.0


def test_vertical_angle():
    angle = (
        CourtLineDetector
        .calculate_line_angle(
            10,
            0,
            10,
            100,
        )
    )

    assert angle == 90.0


def test_horizontal_classification():
    detector = CourtLineDetector()

    assert (
        detector.classify_orientation(
            5.0
        )
        == "horizontal"
    )


def test_vertical_classification():
    detector = CourtLineDetector()

    assert (
        detector.classify_orientation(
            88.0
        )
        == "vertical"
    )


def test_diagonal_classification():
    detector = CourtLineDetector()

    assert (
        detector.classify_orientation(
            45.0
        )
        == "diagonal"
    )


def test_detect_synthetic_lines():
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    cv_color = (
        255,
        255,
        255,
    )

    import cv2

    cv2.line(
        frame,
        (100, 200),
        (540, 200),
        cv_color,
        5,
    )

    cv2.line(
        frame,
        (200, 100),
        (200, 400),
        cv_color,
        5,
    )

    detector = CourtLineDetector(
        CourtLineConfig(
            white_threshold=150,
            hough_threshold=30,
            min_line_length=50,
            max_line_gap=10,
            roi_top_ratio=0.0,
        )
    )

    lines = detector.detect_lines(
        frame
    )

    assert len(lines) > 0