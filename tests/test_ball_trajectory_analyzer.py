import pytest

from backend.app.analytics.ball_trajectory_analyzer import (
    BallTrajectoryAnalysisError,
    BallTrajectoryAnalyzer,
    BallTrajectoryConfig,
)


def make_point(
    frame,
    time,
    x,
    y,
    predicted=False,
):
    return {
        "frame_number": frame,
        "timestamp_seconds": time,
        "x": x,
        "y": y,
        "predicted": predicted,
    }


def test_distance():
    distance = (
        BallTrajectoryAnalyzer
        .calculate_distance(
            (0, 0),
            (3, 4),
        )
    )

    assert distance == 5.0


def test_velocity():
    point_a = make_point(
        0,
        0.0,
        10,
        10,
    )

    point_b = make_point(
        30,
        1.0,
        20,
        30,
    )

    velocity = (
        BallTrajectoryAnalyzer
        .calculate_velocity(
            point_a,
            point_b,
        )
    )

    assert velocity == (
        10.0,
        20.0,
    )


def test_speed():
    speed = (
        BallTrajectoryAnalyzer
        .calculate_speed(
            (
                3,
                4,
            )
        )
    )

    assert speed == 5.0


def test_direction_change():
    angle = (
        BallTrajectoryAnalyzer
        .calculate_direction_change(
            (
                1,
                0,
            ),
            (
                0,
                1,
            ),
        )
    )

    assert angle == pytest.approx(
        90.0,
        abs=0.01,
    )


def test_vertical_reversal():
    result = (
        BallTrajectoryAnalyzer
        .has_vertical_reversal(
            (
                10,
                20,
            ),
            (
                15,
                -10,
            ),
        )
    )

    assert result is True


def test_no_vertical_reversal():
    result = (
        BallTrajectoryAnalyzer
        .has_vertical_reversal(
            (
                10,
                20,
            ),
            (
                15,
                10,
            ),
        )
    )

    assert result is False


def test_prediction_ratio():
    points = [
        make_point(
            1,
            0.1,
            10,
            10,
            False,
        ),
        make_point(
            2,
            0.2,
            20,
            20,
            True,
        ),
    ]

    ratio = (
        BallTrajectoryAnalyzer
        .calculate_prediction_ratio(
            points
        )
    )

    assert ratio == 0.5


def test_bounce_candidate():
    analyzer = (
        BallTrajectoryAnalyzer()
    )

    previous_point = make_point(
        1,
        0.0,
        100,
        100,
    )

    current_point = make_point(
        2,
        0.1,
        110,
        130,
    )

    next_point = make_point(
        3,
        0.2,
        120,
        100,
    )

    result = (
        analyzer.calculate_bounce_score(
            previous_point,
            current_point,
            next_point,
        )
    )

    assert (
        result[
            "vertical_reversal"
        ]
        is True
    )

    assert (
        result[
            "bounce_score"
        ]
        > 0
    )


def test_short_trajectory():
    analyzer = (
        BallTrajectoryAnalyzer()
    )

    result = (
        analyzer.analyse_trajectory(
            [
                make_point(
                    1,
                    0.0,
                    10,
                    10,
                ),
                make_point(
                    2,
                    0.1,
                    20,
                    20,
                ),
            ]
        )
    )

    assert (
        result[
            "analysed_windows"
        ]
        == 0
    )


def test_deduplicate_bounces():
    candidates = [
        {
            "frame_number": 10,
            "bounce_score": 0.6,
        },
        {
            "frame_number": 12,
            "bounce_score": 0.9,
        },
        {
            "frame_number": 30,
            "bounce_score": 0.8,
        },
    ]

    result = (
        BallTrajectoryAnalyzer
        .deduplicate_bounces(
            candidates,
            minimum_frame_gap=4,
        )
    )

    assert len(result) == 2

    assert (
        result[0][
            "frame_number"
        ]
        == 12
    )


def test_reset():
    analyzer = (
        BallTrajectoryAnalyzer()
    )

    analyzer.motion_history.append(
        {
            "test": True
        }
    )

    analyzer.bounce_candidates.append(
        {
            "test": True
        }
    )

    analyzer.reset()

    assert (
        analyzer.motion_history
        == []
    )

    assert (
        analyzer.bounce_candidates
        == []
    )


def test_invalid_config():
    config = (
        BallTrajectoryConfig(
            minimum_points_for_analysis=2
        )
    )

    with pytest.raises(
        BallTrajectoryAnalysisError
    ):
        BallTrajectoryAnalyzer(
            config
        )