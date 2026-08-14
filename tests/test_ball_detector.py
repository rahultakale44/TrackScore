import numpy as np
import pytest

from backend.app.vision.ball_detector import (
    BallDetectionError,
    BallDetector,
    BallDetectorConfig,
)


def create_detector_without_model():
    detector = object.__new__(
        BallDetector
    )

    detector.config = (
        BallDetectorConfig()
    )

    return detector


def test_invalid_confidence():
    detector = (
        create_detector_without_model()
    )

    detector.config = (
        BallDetectorConfig(
            confidence_threshold=1.5
        )
    )

    with pytest.raises(
        BallDetectionError
    ):
        detector._validate_config()


def test_validate_none_frame():
    with pytest.raises(
        BallDetectionError
    ):
        BallDetector.validate_frame(
            None
        )


def test_validate_empty_frame():
    frame = np.array([])

    with pytest.raises(
        BallDetectionError
    ):
        BallDetector.validate_frame(
            frame
        )


def test_filter_candidates():
    detector = (
        create_detector_without_model()
    )

    candidates = [
        {
            "bbox": {
                "area": 30.0,
            },
            "area_ratio": 0.0005,
            "aspect_ratio": 1.0,
        },
        {
            "bbox": {
                "area": 100000.0,
            },
            "area_ratio": 0.5,
            "aspect_ratio": 1.0,
        },
    ]

    result = (
        detector.filter_candidates(
            candidates
        )
    )

    assert len(result) == 1


def test_color_score_yellow():
    detector = (
        create_detector_without_model()
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    frame[
        40:60,
        40:60,
    ] = (
        0,
        255,
        255,
    )

    candidate = {
        "bbox": {
            "x1": 40,
            "y1": 40,
            "x2": 60,
            "y2": 60,
        }
    }

    score = (
        detector.calculate_color_score(
            frame,
            candidate,
        )
    )

    assert score > 0.5


def test_rank_candidates():
    detector = (
        create_detector_without_model()
    )

    detector.config = (
        BallDetectorConfig(
            use_color_score=False
        )
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    candidates = [
        {
            "confidence": 0.3,
            "bbox": {
                "x1": 1,
                "y1": 1,
                "x2": 5,
                "y2": 5,
            },
        },
        {
            "confidence": 0.9,
            "bbox": {
                "x1": 10,
                "y1": 10,
                "x2": 15,
                "y2": 15,
            },
        },
    ]

    ranked = (
        detector.rank_candidates(
            frame,
            candidates,
        )
    )

    assert (
        ranked[0][
            "confidence"
        ]
        == 0.9
    )


def test_draw_no_ball():
    frame = np.zeros(
        (200, 300, 3),
        dtype=np.uint8,
    )

    result = {
        "ball": None
    }

    overlay = (
        BallDetector.draw_detection(
            frame,
            result,
        )
    )

    assert (
        overlay.shape
        == frame.shape
    )