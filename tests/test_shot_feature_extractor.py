import pytest

from backend.app.ml.shot_feature_extractor import (
    ShotFeatureExtractionError,
    ShotFeatureExtractor,
)


def point(
    frame,
    time,
    x,
    y,
    predicted=False,
):
    return {
        "frame_number": (
            frame
        ),
        "timestamp_seconds": (
            time
        ),
        "x": x,
        "y": y,
        "predicted": (
            predicted
        ),
    }


def test_distance():
    result = (
        ShotFeatureExtractor
        .calculate_distance(
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

    assert result == 5.0


def test_angle():
    result = (
        ShotFeatureExtractor
        .calculate_angle(
            point(
                0,
                0,
                0,
                0,
            ),
            point(
                1,
                1,
                10,
                0,
            ),
        )
    )

    assert result == pytest.approx(
        0.0,
        abs=0.01,
    )


def test_angle_difference():
    result = (
        ShotFeatureExtractor
        .normalize_angle_difference(
            170,
            -170,
        )
    )

    assert result == pytest.approx(
        20,
        abs=0.01,
    )


def test_segment_rallies():
    extractor = (
        ShotFeatureExtractor()
    )

    trajectory = [
        point(
            0,
            0.0,
            0,
            0,
        ),
        point(
            1,
            0.1,
            10,
            10,
        ),
        point(
            2,
            0.2,
            20,
            20,
        ),
        point(
            3,
            0.3,
            30,
            30,
        ),

        point(
            40,
            3.0,
            0,
            0,
        ),
        point(
            41,
            3.1,
            10,
            10,
        ),
        point(
            42,
            3.2,
            20,
            20,
        ),
        point(
            43,
            3.3,
            30,
            30,
        ),
    ]

    rallies = (
        extractor
        .segment_rallies(
            trajectory
        )
    )

    assert len(
        rallies
    ) == 2


def test_segment_shots():
    extractor = (
        ShotFeatureExtractor()
    )

    rally = [
        point(
            0,
            0.0,
            0,
            0,
        ),
        point(
            1,
            0.1,
            10,
            0,
        ),
        point(
            2,
            0.2,
            20,
            0,
        ),
        point(
            3,
            0.3,
            20,
            20,
        ),
        point(
            4,
            0.4,
            20,
            30,
        ),
    ]

    shots = (
        extractor
        .segment_shots(
            rally
        )
    )

    assert len(
        shots
    ) >= 1


def test_feature_extraction():
    extractor = (
        ShotFeatureExtractor()
    )

    shot = [
        point(
            0,
            0.0,
            0,
            0,
        ),
        point(
            1,
            0.1,
            10,
            0,
        ),
        point(
            2,
            0.2,
            20,
            0,
        ),
    ]

    result = (
        extractor
        .extract_shot_features(
            shot=shot,
            rally_id=1,
            shot_id=1,
        )
    )

    assert (
        result[
            "rally_id"
        ]
        == 1
    )

    assert (
        result[
            "shot_id"
        ]
        == 1
    )

    assert (
        result[
            "trajectory_distance_pixels"
        ]
        > 0
    )


def test_prediction_ratio():
    extractor = (
        ShotFeatureExtractor()
    )

    shot = [
        point(
            0,
            0.0,
            0,
            0,
            False,
        ),
        point(
            1,
            0.1,
            10,
            0,
            True,
        ),
    ]

    result = (
        extractor
        .extract_shot_features(
            shot=shot,
            rally_id=1,
            shot_id=1,
        )
    )

    assert (
        result[
            "predicted_point_ratio"
        ]
        == 0.5
    )


def test_empty_dataset():
    extractor = (
        ShotFeatureExtractor()
    )

    result = (
        extractor
        .build_feature_dataset(
            []
        )
    )

    assert (
        result[
            "rally_count"
        ]
        == 0
    )

    assert (
        result[
            "shot_count"
        ]
        == 0
    )