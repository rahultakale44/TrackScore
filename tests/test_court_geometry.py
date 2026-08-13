import pytest

from backend.app.vision.court_geometry import (
    CourtGeometry,
    CourtGeometryError,
    TennisCourtDimensions,
)


def test_default_court_dimensions():
    geometry = CourtGeometry()

    singles = geometry.get_dimensions(
        "singles"
    )

    doubles = geometry.get_dimensions(
        "doubles"
    )

    assert (
        singles["length_meters"]
        == 23.77
    )

    assert (
        singles["width_meters"]
        == 8.23
    )

    assert (
        doubles["width_meters"]
        == 10.97
    )


def test_invalid_court_type():
    geometry = CourtGeometry()

    with pytest.raises(
        CourtGeometryError
    ):
        geometry.get_dimensions(
            "invalid"
        )


def test_invalid_dimensions():
    dimensions = TennisCourtDimensions(
        court_length=-1
    )

    with pytest.raises(
        CourtGeometryError
    ):
        CourtGeometry(
            dimensions
        )


def test_keypoints_exist():
    geometry = CourtGeometry()

    points = (
        geometry.get_doubles_keypoints()
    )

    assert "far_left_corner" in points

    assert "far_right_corner" in points

    assert "near_left_corner" in points

    assert "near_right_corner" in points

    assert "court_center" in points


def test_far_left_corner():
    geometry = CourtGeometry()

    points = (
        geometry.get_doubles_keypoints()
    )

    assert (
        points["far_left_corner"]
        == (0.0, 0.0)
    )


def test_near_right_corner():
    geometry = CourtGeometry()

    points = (
        geometry.get_doubles_keypoints()
    )

    assert (
        points["near_right_corner"]
        == (10.97, 23.77)
    )


def test_court_center():
    geometry = CourtGeometry()

    points = (
        geometry.get_doubles_keypoints()
    )

    center_x, center_y = (
        points["court_center"]
    )

    assert center_x == pytest.approx(
        10.97 / 2
    )

    assert center_y == pytest.approx(
        23.77 / 2
    )


def test_inside_singles_court():
    geometry = CourtGeometry()

    assert (
        geometry.is_point_inside_court(
            4.0,
            10.0,
            "singles",
        )
        is True
    )


def test_outside_singles_court():
    geometry = CourtGeometry()

    assert (
        geometry.is_point_inside_court(
            9.0,
            10.0,
            "singles",
        )
        is False
    )


def test_inside_doubles_court():
    geometry = CourtGeometry()

    assert (
        geometry.is_point_inside_court(
            10.0,
            10.0,
            "doubles",
        )
        is True
    )


def test_zone_classification():
    geometry = CourtGeometry()

    zone = geometry.classify_court_zone(
        2.0,
        2.0,
        "singles",
    )

    assert zone == "far_backcourt_left"


def test_out_zone():
    geometry = CourtGeometry()

    zone = geometry.classify_court_zone(
        -1.0,
        5.0,
        "singles",
    )

    assert zone == "out"


def test_court_lines():
    geometry = CourtGeometry()

    lines = geometry.get_court_lines()

    assert len(lines) > 0

    names = [
        line[0]
        for line in lines
    ]

    assert "net" in names

    assert "far_baseline" in names

    assert "near_baseline" in names