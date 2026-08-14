import numpy as np
import pytest

from backend.app.vision.ball_tracker import (
    BallTracker,
    BallTrackerConfig,
    BallTrackingError,
)


def create_tracker_without_model():
    tracker = object.__new__(
        BallTracker
    )

    tracker.config = (
        BallTrackerConfig()
    )

    tracker.last_position = None

    tracker.last_timestamp = None

    tracker.velocity = (
        0.0,
        0.0,
    )

    tracker.missed_frames = 0

    tracker.history = []

    return tracker


def test_invalid_match_distance():
    tracker = (
        create_tracker_without_model()
    )

    tracker.config = (
        BallTrackerConfig(
            maximum_match_distance_pixels=0
        )
    )

    with pytest.raises(
        BallTrackingError
    ):
        tracker._validate_config()


def test_validate_none_frame():
    with pytest.raises(
        BallTrackingError
    ):
        BallTracker.validate_frame(
            None
        )


def test_validate_valid_frame():
    frame = np.zeros(
        (
            200,
            300,
            3,
        ),
        dtype=np.uint8,
    )

    BallTracker.validate_frame(
        frame
    )


def test_distance():
    distance = (
        BallTracker.calculate_distance(
            (
                0,
                0,
            ),
            (
                3,
                4,
            ),
        )
    )

    assert distance == 5.0


def test_no_prediction_without_position():
    tracker = (
        create_tracker_without_model()
    )

    prediction = (
        tracker.predict_position(
            1.0
        )
    )

    assert prediction is None


def test_position_prediction():
    tracker = (
        create_tracker_without_model()
    )

    tracker.last_position = (
        100.0,
        100.0,
    )

    tracker.last_timestamp = 1.0

    tracker.velocity = (
        20.0,
        10.0,
    )

    prediction = (
        tracker.predict_position(
            2.0
        )
    )

    assert (
        prediction[0]
        == pytest.approx(
            120.0
        )
    )

    assert (
        prediction[1]
        == pytest.approx(
            110.0
        )
    )


def test_candidate_selection_without_prediction():
    tracker = (
        create_tracker_without_model()
    )

    candidates = [
        {
            "center": {
                "x": 50,
                "y": 60,
            },
            "ranking_score": 0.9,
        }
    ]

    result = (
        tracker.select_temporal_candidate(
            candidates,
            None,
        )
    )

    assert result is not None


def test_candidate_near_prediction():
    tracker = (
        create_tracker_without_model()
    )

    candidates = [
        {
            "center": {
                "x": 105,
                "y": 100,
            },
            "ranking_score": 0.7,
        },

        {
            "center": {
                "x": 400,
                "y": 400,
            },
            "ranking_score": 0.99,
        },
    ]

    selected = (
        tracker.select_temporal_candidate(
            candidates,
            (
                100,
                100,
            ),
        )
    )

    assert selected is not None

    assert (
        selected[
            "center"
        ]["x"]
        == 105
    )


def test_position_smoothing_first_point():
    tracker = (
        create_tracker_without_model()
    )

    result = (
        tracker.smooth_position(
            (
                100,
                200,
            )
        )
    )

    assert result == (
        100,
        200,
    )


def test_position_smoothing():
    tracker = (
        create_tracker_without_model()
    )

    tracker.last_position = (
        100.0,
        100.0,
    )

    result = (
        tracker.smooth_position(
            (
                110.0,
                110.0,
            )
        )
    )

    assert (
        result[0]
        > 100.0
    )

    assert (
        result[0]
        < 110.0
    )


def test_history_limit():
    tracker = (
        create_tracker_without_model()
    )

    tracker.config = (
        BallTrackerConfig(
            trajectory_length=3
        )
    )

    for index in range(
        5
    ):
        tracker.add_to_history(
            position=(
                index,
                index,
            ),
            frame_number=index,
            timestamp_seconds=(
                index / 30
            ),
            source="test",
        )

    assert (
        len(
            tracker.history
        )
        == 3
    )


def test_missing_detection_prediction():
    tracker = (
        create_tracker_without_model()
    )

    tracker.last_position = (
        100.0,
        100.0,
    )

    tracker.last_timestamp = 0.0

    tracker.velocity = (
        30.0,
        0.0,
    )

    result = (
        tracker.handle_missing_detection(
            frame_number=1,
            timestamp_seconds=1.0,
        )
    )

    assert result is not None

    assert (
        result[
            "predicted"
        ]
        is True
    )


def test_reset():
    tracker = (
        create_tracker_without_model()
    )

    tracker.last_position = (
        10,
        20,
    )

    tracker.velocity = (
        5,
        5,
    )

    tracker.history.append(
        {
            "x": 10,
            "y": 20,
        }
    )

    tracker.reset()

    assert (
        tracker.last_position
        is None
    )

    assert (
        tracker.velocity
        == (
            0.0,
            0.0,
        )
    )

    assert (
        tracker.history
        == []
    )