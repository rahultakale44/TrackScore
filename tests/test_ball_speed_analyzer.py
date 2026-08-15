import pytest

from backend.app.analytics.ball_speed_analyzer import (
    BallSpeedAnalysisError,
    BallSpeedAnalyzer,
    BallSpeedConfig,
)

from backend.app.vision.court_homography import (
    CourtHomography,
)


@pytest.fixture
def homography():
    homography = CourtHomography()

    homography.calibrate(
        [
            [100, 100],
            [500, 100],
            [100, 700],
            [500, 700],
        ],
        court_type="singles",
    )

    return homography


@pytest.fixture
def analyzer(
    homography,
):
    return BallSpeedAnalyzer(
        homography
    )


def test_requires_calibration():
    homography = (
        CourtHomography()
    )

    with pytest.raises(
        BallSpeedAnalysisError
    ):
        BallSpeedAnalyzer(
            homography
        )


def test_distance():
    result = (
        BallSpeedAnalyzer
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


def test_mps_to_kmh():
    result = (
        BallSpeedAnalyzer
        .meters_per_second_to_kmh(
            10.0
        )
    )

    assert result == 36.0


def test_speed(
    analyzer,
):
    result = (
        analyzer.calculate_speed(
            distance_meters=10.0,
            delta_time_seconds=1.0,
        )
    )

    assert result == 36.0


def test_unreasonable_speed_removed(
    analyzer,
):
    result = (
        analyzer.calculate_speed(
            distance_meters=100,
            delta_time_seconds=0.1,
        )
    )

    assert result == 0.0


def test_pixel_mapping(
    analyzer,
):
    x, y = (
        analyzer.pixel_to_court(
            100,
            100,
        )
    )

    assert x == pytest.approx(
        0.0,
        abs=0.01,
    )

    assert y == pytest.approx(
        0.0,
        abs=0.01,
    )


def test_first_point(
    analyzer,
):
    result = (
        analyzer.analyse_point(
            {
                "frame_number": 0,
                "timestamp_seconds": 0.0,
                "x": 100,
                "y": 100,
                "predicted": False,
            }
        )
    )

    assert (
        result[
            "current_speed_kmh"
        ]
        == 0.0
    )


def test_two_points(
    analyzer,
):
    analyzer.analyse_point(
        {
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "x": 100,
            "y": 100,
            "predicted": False,
        }
    )

    result = (
        analyzer.analyse_point(
            {
                "frame_number": 30,
                "timestamp_seconds": 1.0,
                "x": 150,
                "y": 100,
                "predicted": False,
            }
        )
    )

    assert (
        result[
            "total_distance_meters"
        ]
        > 0
    )


def test_predicted_point_ignored(
    analyzer,
):
    analyzer.analyse_point(
        {
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "x": 100,
            "y": 100,
            "predicted": False,
        }
    )

    result = (
        analyzer.analyse_point(
            {
                "frame_number": 1,
                "timestamp_seconds": 0.1,
                "x": 400,
                "y": 400,
                "predicted": True,
            }
        )
    )

    assert (
        result[
            "current_speed_kmh"
        ]
        == 0.0
    )


def test_summary(
    analyzer,
):
    result = (
        analyzer.get_summary()
    )

    assert (
        "peak_speed_kmh"
        in result
    )

    assert (
        result[
            "speed_type"
        ]
        == "court-plane estimated speed"
    )


def test_reset(
    analyzer,
):
    analyzer.speed_history.append(
        20
    )

    analyzer.total_distance_meters = (
        12
    )

    analyzer.reset()

    assert (
        analyzer.speed_history
        == []
    )

    assert (
        analyzer.total_distance_meters
        == 0.0
    )


def test_invalid_config(
    homography,
):
    config = (
        BallSpeedConfig(
            smoothing_window=0
        )
    )

    with pytest.raises(
        BallSpeedAnalysisError
    ):
        BallSpeedAnalyzer(
            homography,
            config,
        )