import numpy as np
import pytest

from backend.app.vision.player_tracker import (
    PlayerTracker,
    PlayerTrackerConfig,
    PlayerTrackingError,
)


def make_tracker_without_model():
    tracker = object.__new__(
        PlayerTracker
    )

    tracker.config = (
        PlayerTrackerConfig()
    )

    tracker.track_to_player_label = {}

    tracker.player_history = {
        "Player A": [],
        "Player B": [],
    }

    return tracker


def test_invalid_confidence():
    tracker = make_tracker_without_model()

    tracker.config = PlayerTrackerConfig(
        confidence_threshold=1.5
    )

    with pytest.raises(
        PlayerTrackingError
    ):
        tracker._validate_config()


def test_validate_none_frame():
    with pytest.raises(
        PlayerTrackingError
    ):
        PlayerTracker.validate_frame(
            None
        )


def test_validate_valid_frame():
    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    PlayerTracker.validate_frame(
        frame
    )


def test_selection_score():
    track = {
        "confidence": 0.9,
        "bbox": {
            "width": 50.0,
            "height": 100.0,
        },
    }

    score = (
        PlayerTracker
        ._selection_score(
            track
        )
    )

    assert score > 0


def test_stable_player_labels():
    tracker = (
        make_tracker_without_model()
    )

    tracks = [
        {
            "track_id": 5,
            "foot_point": {
                "x": 300,
                "y": 300,
            },
        },
        {
            "track_id": 9,
            "foot_point": {
                "x": 400,
                "y": 700,
            },
        },
    ]

    labelled = (
        tracker.assign_stable_labels(
            tracks
        )
    )

    assert (
        labelled[0][
            "player_label"
        ]
        == "Player A"
    )

    assert (
        labelled[1][
            "player_label"
        ]
        == "Player B"
    )


def test_labels_remain_stable():
    tracker = (
        make_tracker_without_model()
    )

    first_frame = [
        {
            "track_id": 10,
            "foot_point": {
                "x": 300,
                "y": 300,
            },
        },
        {
            "track_id": 20,
            "foot_point": {
                "x": 400,
                "y": 700,
            },
        },
    ]

    tracker.assign_stable_labels(
        first_frame
    )

    second_frame = [
        {
            "track_id": 20,
            "foot_point": {
                "x": 410,
                "y": 690,
            },
        },
        {
            "track_id": 10,
            "foot_point": {
                "x": 310,
                "y": 320,
            },
        },
    ]

    labelled = (
        tracker.assign_stable_labels(
            second_frame
        )
    )

    labels = {
        item["track_id"]:
        item["player_label"]
        for item in labelled
    }

    assert labels[10] == "Player A"
    assert labels[20] == "Player B"


def test_history_update():
    tracker = (
        make_tracker_without_model()
    )

    players = [
        {
            "track_id": 5,
            "player_label": (
                "Player A"
            ),
            "frame_number": 10,
            "timestamp_seconds": 0.33,
            "center": {
                "x": 100,
                "y": 200,
            },
            "foot_point": {
                "x": 100,
                "y": 250,
            },
            "confidence": 0.95,
        }
    ]

    tracker.update_history(
        players
    )

    assert (
        len(
            tracker.player_history[
                "Player A"
            ]
        )
        == 1
    )


def test_reset():
    tracker = (
        make_tracker_without_model()
    )

    tracker.track_to_player_label[
        5
    ] = "Player A"

    tracker.player_history[
        "Player A"
    ].append(
        {
            "frame_number": 1
        }
    )

    tracker.reset()

    assert (
        tracker.track_to_player_label
        == {}
    )

    assert (
        tracker.player_history[
            "Player A"
        ]
        == []
    )