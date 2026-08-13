import numpy as np
import pytest

from backend.app.vision.court_homography import (
    CourtHomography,
    CourtHomographyError,
)


@pytest.fixture
def image_corners():
    return [
        [100.0, 100.0],
        [500.0, 100.0],
        [100.0, 700.0],
        [500.0, 700.0],
    ]


@pytest.fixture
def calibrated_homography(
    image_corners,
):
    homography = CourtHomography()

    homography.calibrate(
        image_corners,
        court_type="singles",
    )

    return homography


def test_not_calibrated():
    homography = CourtHomography()

    assert (
        homography.is_calibrated()
        is False
    )


def test_calibration(
    image_corners,
):
    homography = CourtHomography()

    calibration = (
        homography.calibrate(
            image_corners,
            "singles",
        )
    )

    assert (
        calibration.matrix.shape
        == (3, 3)
    )

    assert (
        calibration.inverse_matrix.shape
        == (3, 3)
    )

    assert (
        homography.is_calibrated()
        is True
    )


def test_invalid_point_count():
    homography = CourtHomography()

    points = [
        [1, 1],
        [2, 2],
        [3, 3],
    ]

    with pytest.raises(
        CourtHomographyError
    ):
        homography.calibrate(
            points
        )


def test_far_left_mapping(
    calibrated_homography,
):
    court_point = (
        calibrated_homography
        .image_to_court(
            (100, 100)
        )
    )

    assert court_point[0] == pytest.approx(
        0.0,
        abs=0.001,
    )

    assert court_point[1] == pytest.approx(
        0.0,
        abs=0.001,
    )


def test_far_right_mapping(
    calibrated_homography,
):
    court_point = (
        calibrated_homography
        .image_to_court(
            (500, 100)
        )
    )

    assert court_point[0] == pytest.approx(
        8.23,
        abs=0.001,
    )

    assert court_point[1] == pytest.approx(
        0.0,
        abs=0.001,
    )


def test_near_right_mapping(
    calibrated_homography,
):
    court_point = (
        calibrated_homography
        .image_to_court(
            (500, 700)
        )
    )

    assert court_point[0] == pytest.approx(
        8.23,
        abs=0.001,
    )

    assert court_point[1] == pytest.approx(
        23.77,
        abs=0.001,
    )


def test_court_to_image(
    calibrated_homography,
):
    image_point = (
        calibrated_homography
        .court_to_image(
            (0.0, 0.0)
        )
    )

    assert image_point[0] == pytest.approx(
        100.0,
        abs=0.01,
    )

    assert image_point[1] == pytest.approx(
        100.0,
        abs=0.01,
    )


def test_multiple_image_points(
    calibrated_homography,
):
    transformed = (
        calibrated_homography
        .image_points_to_court(
            [
                [100, 100],
                [500, 700],
            ]
        )
    )

    assert len(transformed) == 2


def test_point_inside_court(
    calibrated_homography,
):
    analysis = (
        calibrated_homography
        .analyse_image_point(
            (300, 400)
        )
    )

    assert (
        analysis["inside_court"]
        is True
    )


def test_real_distance():
    distance = (
        CourtHomography
        .calculate_real_distance(
            (0, 0),
            (3, 4),
        )
    )

    assert distance == 5.0


def test_uncalibrated_conversion():
    homography = CourtHomography()

    with pytest.raises(
        CourtHomographyError
    ):
        homography.image_to_court(
            (100, 100)
        )


def test_custom_correspondences():
    homography = CourtHomography()

    image_points = [
        [100, 100],
        [500, 100],
        [100, 700],
        [500, 700],
        [300, 400],
    ]

    real_points = [
        [0.0, 0.0],
        [8.23, 0.0],
        [0.0, 23.77],
        [8.23, 23.77],
        [4.115, 11.885],
    ]

    calibration = (
        homography
        .calibrate_from_correspondences(
            image_points,
            real_points,
            court_type="singles",
        )
    )

    assert (
        calibration.matrix.shape
        == (3, 3)
    )


def test_calibration_summary(
    calibrated_homography,
):
    summary = (
        calibrated_homography
        .get_calibration_summary()
    )

    assert (
        summary["calibrated"]
        is True
    )

    assert (
        summary["court_type"]
        == "singles"
    )

    assert (
        summary["court_width_meters"]
        == 8.23
    )

    assert (
        summary["court_length_meters"]
        == 23.77
    )