import pytest

from backend.app.analytics.player_motion_analyzer import (
    PlayerMotionAnalysisError,
    PlayerMotionAnalyzer,
    PlayerMotionConfig,
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
    return PlayerMotionAnalyzer(
        homography
    )


def test_requires_calibrated_homography():
    homography = CourtHomography()

    with pytest.raises(
        PlayerMotionAnalysisError
    ):
        PlayerMotionAnalyzer(
            homography
        )


def test_distance():
    distance = (
        PlayerMotionAnalyzer
        .calculate_distance(
            (0, 0),
            (3, 4),
        )
    )

    assert distance == 5.0


def test_mps_to_kmh():
    speed = (
        PlayerMotionAnalyzer
        .meters_per_second_to_kmh(
            10.0
        )
    )

    assert speed == 36.0


def test_speed_calculation(
    analyzer,
):
    speed = (
        analyzer.calculate_speed(
            distance_meters=2.0,
            time_delta_seconds=1.0,
        )
    )

    assert speed == 7.2


def test_unreasonable_speed_removed(
    analyzer,
):
    speed = (
        analyzer.calculate_speed(
            distance_meters=20,
            time_delta_seconds=0.1,
        )
    )

    assert speed == 0.0


def test_player_pixel_to_court(
    analyzer,
):
    player = {
        "player_label": (
            "Player A"
        ),
        "foot_point": {
            "x": 100,
            "y": 100,
        },
        "timestamp_seconds": 0.0,
        "frame_number": 0,
    }

    point = (
        analyzer
        .player_pixel_to_court(
            player
        )
    )

    assert point[0] == pytest.approx(
        0.0,
        abs=0.01,
    )

    assert point[1] == pytest.approx(
        0.0,
        abs=0.01,
    )


def test_first_player_sample(
    analyzer,
):
    player = {
        "track_id": 1,
        "player_label": (
            "Player A"
        ),
        "foot_point": {
            "x": 300,
            "y": 400,
        },
        "timestamp_seconds": 0.0,
        "frame_number": 0,
    }

    result = (
        analyzer
        .analyse_player(
            player
        )
    )

    assert (
        result[
            "movement"
        ][
            "total_distance_meters"
        ]
        == 0.0
    )


def test_two_player_samples(
    analyzer,
):
    first = {
        "track_id": 1,
        "player_label": (
            "Player A"
        ),
        "foot_point": {
            "x": 200,
            "y": 400,
        },
        "timestamp_seconds": 0.0,
        "frame_number": 0,
    }

    second = {
        "track_id": 1,
        "player_label": (
            "Player A"
        ),
        "foot_point": {
            "x": 210,
            "y": 400,
        },
        "timestamp_seconds": 1.0,
        "frame_number": 30,
    }

    analyzer.analyse_player(
        first
    )

    result = (
        analyzer
        .analyse_player(
            second
        )
    )

    assert (
        result[
            "movement"
        ][
            "total_distance_meters"
        ]
        > 0
    )


def test_summary(
    analyzer,
):
    player = {
        "track_id": 1,
        "player_label": (
            "Player A"
        ),
        "foot_point": {
            "x": 300,
            "y": 400,
        },
        "timestamp_seconds": 0,
        "frame_number": 0,
    }

    analyzer.analyse_player(
        player
    )

    summary = (
        analyzer.get_summary()
    )

    assert (
        "Player A"
        in summary
    )


def test_reset(
    analyzer,
):
    player = {
        "track_id": 1,
        "player_label": (
            "Player A"
        ),
        "foot_point": {
            "x": 300,
            "y": 400,
        },
        "timestamp_seconds": 0,
        "frame_number": 0,
    }

    analyzer.analyse_player(
        player
    )

    analyzer.reset()

    assert (
        analyzer.player_state
        == {}
    )


def test_invalid_config(
    homography,
):
    config = PlayerMotionConfig(
        smoothing_window=0
    )

    with pytest.raises(
        PlayerMotionAnalysisError
    ):
        PlayerMotionAnalyzer(
            homography,
            config,
        )