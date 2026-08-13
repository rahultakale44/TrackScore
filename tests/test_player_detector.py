import numpy as np
import pytest

from backend.app.vision.player_detector import (
    PlayerDetectionError,
    PlayerDetector,
    PlayerDetectorConfig,
)


def test_invalid_confidence():
    config = PlayerDetectorConfig(
        confidence_threshold=1.5
    )

    detector = object.__new__(
        PlayerDetector
    )

    detector.config = config

    with pytest.raises(
        PlayerDetectionError
    ):
        detector._validate_config()


def test_invalid_max_players():
    config = PlayerDetectorConfig(
        max_players=0
    )

    detector = object.__new__(
        PlayerDetector
    )

    detector.config = config

    with pytest.raises(
        PlayerDetectionError
    ):
        detector._validate_config()


def test_validate_none_frame():
    with pytest.raises(
        PlayerDetectionError
    ):
        PlayerDetector.validate_frame(
            None
        )


def test_validate_empty_frame():
    frame = np.array([])

    with pytest.raises(
        PlayerDetectionError
    ):
        PlayerDetector.validate_frame(
            frame
        )


def test_player_score():
    detection = {
        "confidence": 0.9,
        "bbox": {
            "width": 50.0,
            "height": 100.0,
        },
    }

    score = (
        PlayerDetector
        ._calculate_player_score(
            detection
        )
    )

    assert score > 0


def test_assign_two_players():
    players = [
        {
            "confidence": 0.9,
            "foot_point": {
                "x": 300,
                "y": 700,
            },
        },
        {
            "confidence": 0.8,
            "foot_point": {
                "x": 400,
                "y": 300,
            },
        },
    ]

    labelled = (
        PlayerDetector
        .assign_player_labels(
            players
        )
    )

    assert (
        labelled[0][
            "player_label"
        ]
        == "Player A"
    )

    assert (
        labelled[0][
            "court_side"
        ]
        == "far"
    )

    assert (
        labelled[1][
            "player_label"
        ]
        == "Player B"
    )

    assert (
        labelled[1][
            "court_side"
        ]
        == "near"
    )


def test_empty_player_assignment():
    labelled = (
        PlayerDetector
        .assign_player_labels(
            []
        )
    )

    assert labelled == []


def test_filter_court_people():
    detector = object.__new__(
        PlayerDetector
    )

    detector.config = (
        PlayerDetectorConfig()
    )

    detections = [
        {
            "foot_point": {
                "x": 300,
                "y": 500,
            },
            "bbox": {
                "height": 120,
            },
        },
        {
            "foot_point": {
                "x": 100,
                "y": 50,
            },
            "bbox": {
                "height": 10,
            },
        },
    ]

    filtered = (
        detector.filter_court_people(
            detections,
            (
                720,
                1280,
                3,
            ),
        )
    )

    assert len(filtered) == 1