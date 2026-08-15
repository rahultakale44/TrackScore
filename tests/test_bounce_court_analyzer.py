import pytest

from backend.app.analytics.bounce_court_analyzer import (
    BounceCourtAnalysisError,
    BounceCourtAnalyzer,
    BounceCourtConfig,
)

from backend.app.vision.court_homography import (
    CourtHomography,
)


@pytest.fixture
def homography():
    homography = (
        CourtHomography()
    )

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
    return (
        BounceCourtAnalyzer(
            homography
        )
    )


def make_bounce(
    x,
    y,
    score=0.8,
):
    return {
        "frame_number": 10,
        "timestamp_seconds": 1.0,
        "position": {
            "x": x,
            "y": y,
        },
        "bounce_score": score,
    }


def test_requires_calibration():
    homography = (
        CourtHomography()
    )

    with pytest.raises(
        BounceCourtAnalysisError
    ):
        BounceCourtAnalyzer(
            homography
        )


def test_invalid_court_type(
    homography,
):
    config = (
        BounceCourtConfig(
            court_type="invalid"
        )
    )

    with pytest.raises(
        BounceCourtAnalysisError
    ):
        BounceCourtAnalyzer(
            homography,
            config,
        )


def test_map_far_left(
    analyzer,
):
    bounce = make_bounce(
        100,
        100,
    )

    result = (
        analyzer.map_bounce_to_court(
            bounce
        )
    )

    assert (
        result["x_meters"]
        == pytest.approx(
            0.0,
            abs=0.01,
        )
    )

    assert (
        result["y_meters"]
        == pytest.approx(
            0.0,
            abs=0.01,
        )
    )


def test_inside_call(
    analyzer,
):
    result = (
        analyzer.classify_line_call(
            4.0,
            10.0,
        )
    )

    assert (
        result["call"]
        == "IN"
    )


def test_outside_call(
    analyzer,
):
    result = (
        analyzer.classify_line_call(
            9.0,
            10.0,
        )
    )

    assert (
        result["call"]
        == "OUT"
    )


def test_boundary_tolerance(
    analyzer,
):
    result = (
        analyzer.classify_line_call(
            -0.02,
            10.0,
        )
    )

    assert (
        result["call"]
        == "IN"
    )

    assert (
        result[
            "on_or_near_line"
        ]
        is True
    )


def test_boundary_distances(
    analyzer,
):
    result = (
        analyzer.calculate_boundary_distances(
            4.0,
            10.0,
        )
    )

    assert (
        result[
            "left_sideline_meters"
        ]
        == 4.0
    )


def test_nearest_boundary(
    analyzer,
):
    distances = {
        "left": 0.1,
        "right": 5.0,
        "far": 6.0,
        "near": 4.0,
    }

    result = (
        analyzer.find_nearest_boundary(
            distances
        )
    )

    assert (
        result["boundary"]
        == "left"
    )


def test_complete_bounce_analysis(
    analyzer,
):
    bounce = make_bounce(
        300,
        400,
    )

    result = (
        analyzer.analyse_bounce(
            bounce
        )
    )

    assert (
        result[
            "accepted_bounce"
        ]
        is True
    )

    assert (
        result["line_call"]
        == "IN"
    )


def test_low_score_bounce(
    analyzer,
):
    bounce = make_bounce(
        300,
        400,
        score=0.2,
    )

    result = (
        analyzer.analyse_bounce(
            bounce
        )
    )

    assert (
        result[
            "accepted_bounce"
        ]
        is False
    )


def test_summary():
    bounces = [
        {
            "accepted_bounce": True,
            "line_call": "IN",
            "on_or_near_line": False,
        },
        {
            "accepted_bounce": True,
            "line_call": "OUT",
            "on_or_near_line": False,
        },
        {
            "accepted_bounce": False,
            "line_call": "IN",
            "on_or_near_line": True,
        },
    ]

    summary = (
        BounceCourtAnalyzer
        .summarize(
            bounces
        )
    )

    assert (
        summary[
            "accepted_bounces"
        ]
        == 2
    )

    assert (
        summary[
            "in_bounces"
        ]
        == 1
    )

    assert (
        summary[
            "out_bounces"
        ]
        == 1
    )