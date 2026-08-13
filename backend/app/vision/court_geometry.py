from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


class CourtGeometryError(Exception):
    """Raised when tennis court geometry configuration is invalid."""


@dataclass(frozen=True)
class TennisCourtDimensions:
    """
    Standard tennis court dimensions in meters.
    """

    court_length: float = 23.77

    singles_width: float = 8.23
    doubles_width: float = 10.97

    service_line_distance_from_net: float = 6.40

    net_to_baseline: float = 11.885

    center_service_line_half_length: float = 6.40

    doubles_alley_width: float = 1.37


class CourtGeometry:
    """
    Represents a tennis court in a real-world 2D coordinate system.

    Coordinate system:

        Origin:
            Top-left doubles baseline corner.

        X-axis:
            Court width.

        Y-axis:
            Court length.

    Example:

        (0, 0) -------------------- (10.97, 0)
          |                             |
          |                             |
          |                             |
          |------------- NET -----------|
          |                             |
          |                             |
          |                             |
        (0, 23.77) ---------------- (10.97, 23.77)
    """

    def __init__(
        self,
        dimensions: TennisCourtDimensions | None = None,
    ):
        self.dimensions = (
            dimensions
            if dimensions is not None
            else TennisCourtDimensions()
        )

        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        dimensions = self.dimensions

        if dimensions.court_length <= 0:
            raise CourtGeometryError(
                "Court length must be greater than zero."
            )

        if dimensions.singles_width <= 0:
            raise CourtGeometryError(
                "Singles width must be greater than zero."
            )

        if dimensions.doubles_width <= 0:
            raise CourtGeometryError(
                "Doubles width must be greater than zero."
            )

        if (
            dimensions.doubles_width
            <= dimensions.singles_width
        ):
            raise CourtGeometryError(
                "Doubles court width must be larger than singles width."
            )

        if dimensions.net_to_baseline <= 0:
            raise CourtGeometryError(
                "Net-to-baseline distance must be greater than zero."
            )

    # ============================================================
    # BASIC DIMENSIONS
    # ============================================================

    def get_dimensions(
        self,
        court_type: str = "singles",
    ) -> Dict[str, float]:
        court_type = court_type.lower()

        if court_type == "singles":
            width = self.dimensions.singles_width

        elif court_type == "doubles":
            width = self.dimensions.doubles_width

        else:
            raise CourtGeometryError(
                "court_type must be either 'singles' or 'doubles'."
            )

        return {
            "length_meters": self.dimensions.court_length,
            "width_meters": width,
        }

    # ============================================================
    # STANDARD COURT KEYPOINTS
    # ============================================================

    def get_doubles_keypoints(
        self,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Return important tennis-court landmarks in meters.
        """

        d = self.dimensions

        width = d.doubles_width
        length = d.court_length

        net_y = length / 2.0

        near_service_y = (
            net_y
            + d.service_line_distance_from_net
        )

        far_service_y = (
            net_y
            - d.service_line_distance_from_net
        )

        singles_left = d.doubles_alley_width

        singles_right = (
            width
            - d.doubles_alley_width
        )

        center_x = width / 2.0

        return {
            # Doubles outer corners
            "far_left_corner": (
                0.0,
                0.0,
            ),
            "far_right_corner": (
                width,
                0.0,
            ),
            "near_left_corner": (
                0.0,
                length,
            ),
            "near_right_corner": (
                width,
                length,
            ),

            # Singles baseline corners
            "far_left_singles": (
                singles_left,
                0.0,
            ),
            "far_right_singles": (
                singles_right,
                0.0,
            ),
            "near_left_singles": (
                singles_left,
                length,
            ),
            "near_right_singles": (
                singles_right,
                length,
            ),

            # Net outer points
            "net_left": (
                0.0,
                net_y,
            ),
            "net_right": (
                width,
                net_y,
            ),

            # Singles net points
            "net_left_singles": (
                singles_left,
                net_y,
            ),
            "net_right_singles": (
                singles_right,
                net_y,
            ),

            # Service lines
            "far_service_left": (
                singles_left,
                far_service_y,
            ),
            "far_service_center": (
                center_x,
                far_service_y,
            ),
            "far_service_right": (
                singles_right,
                far_service_y,
            ),

            "near_service_left": (
                singles_left,
                near_service_y,
            ),
            "near_service_center": (
                center_x,
                near_service_y,
            ),
            "near_service_right": (
                singles_right,
                near_service_y,
            ),

            # Court center
            "court_center": (
                center_x,
                net_y,
            ),
        }

    # ============================================================
    # COURT LINES
    # ============================================================

    def get_court_lines(
        self,
    ) -> List[
        Tuple[
            str,
            Tuple[float, float],
            Tuple[float, float],
        ]
    ]:
        """
        Return named court lines using real-world coordinates.
        """

        p = self.get_doubles_keypoints()

        return [
            (
                "far_baseline",
                p["far_left_corner"],
                p["far_right_corner"],
            ),
            (
                "near_baseline",
                p["near_left_corner"],
                p["near_right_corner"],
            ),

            (
                "left_doubles_sideline",
                p["far_left_corner"],
                p["near_left_corner"],
            ),
            (
                "right_doubles_sideline",
                p["far_right_corner"],
                p["near_right_corner"],
            ),

            (
                "left_singles_sideline",
                p["far_left_singles"],
                p["near_left_singles"],
            ),
            (
                "right_singles_sideline",
                p["far_right_singles"],
                p["near_right_singles"],
            ),

            (
                "net",
                p["net_left"],
                p["net_right"],
            ),

            (
                "far_service_line",
                p["far_service_left"],
                p["far_service_right"],
            ),
            (
                "near_service_line",
                p["near_service_left"],
                p["near_service_right"],
            ),

            (
                "center_service_line_far",
                p["net_left_singles"],
                p["far_service_center"],
            ),
            (
                "center_service_line_near",
                p["net_left_singles"],
                p["near_service_center"],
            ),
        ]

    # ============================================================
    # COURT ZONES
    # ============================================================

    def classify_court_zone(
        self,
        x: float,
        y: float,
        court_type: str = "singles",
    ) -> str:
        """
        Classify a real-world court coordinate into a broad zone.
        """

        dimensions = self.get_dimensions(
            court_type
        )

        width = dimensions["width_meters"]
        length = dimensions["length_meters"]

        if (
            x < 0
            or y < 0
            or x > width
            or y > length
        ):
            return "out"

        third = length / 3.0

        if y < third:
            depth = "far_backcourt"

        elif y < 2 * third:
            depth = "midcourt"

        else:
            depth = "near_backcourt"

        half_width = width / 2.0

        if x < half_width:
            side = "left"

        else:
            side = "right"

        return f"{depth}_{side}"

    # ============================================================
    # IN / OUT CHECK
    # ============================================================

    def is_point_inside_court(
        self,
        x: float,
        y: float,
        court_type: str = "singles",
    ) -> bool:
        dimensions = self.get_dimensions(
            court_type
        )

        width = dimensions["width_meters"]
        length = dimensions["length_meters"]

        return (
            0.0 <= x <= width
            and
            0.0 <= y <= length
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(self) -> Dict[str, object]:
        return {
            "court_length_meters": (
                self.dimensions.court_length
            ),
            "singles_width_meters": (
                self.dimensions.singles_width
            ),
            "doubles_width_meters": (
                self.dimensions.doubles_width
            ),
            "net_to_baseline_meters": (
                self.dimensions.net_to_baseline
            ),
            "service_line_distance_from_net_meters": (
                self.dimensions.service_line_distance_from_net
            ),
            "doubles_alley_width_meters": (
                self.dimensions.doubles_alley_width
            ),
            "keypoints": (
                self.get_doubles_keypoints()
            ),
        }