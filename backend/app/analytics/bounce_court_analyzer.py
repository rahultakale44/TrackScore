from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.app.vision.court_homography import (
    CourtHomography,
    CourtHomographyError,
)


class BounceCourtAnalysisError(Exception):
    """Raised when bounce court analysis fails."""


@dataclass
class BounceCourtConfig:
    """
    Configuration for bounce position and line-call analysis.
    """

    court_type: str = "singles"

    boundary_tolerance_meters: float = 0.05

    minimum_bounce_score: float = 0.55


class BounceCourtAnalyzer:
    """
    Converts bounce candidate pixels into real-world court coordinates.

    Responsibilities:
    - Pixel -> meter transformation
    - Court boundary checking
    - IN / OUT classification
    - Court-zone classification
    - Boundary-distance measurement
    """

    def __init__(
        self,
        homography: CourtHomography,
        config: BounceCourtConfig | None = None,
    ):
        if homography is None:
            raise BounceCourtAnalysisError(
                "CourtHomography instance is required."
            )

        if not homography.is_calibrated():
            raise BounceCourtAnalysisError(
                "Court homography must be calibrated first."
            )

        self.homography = homography

        self.config = (
            config
            if config is not None
            else BounceCourtConfig()
        )

        self._validate_config()

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        if self.config.court_type not in {
            "singles",
            "doubles",
        }:
            raise BounceCourtAnalysisError(
                "court_type must be either 'singles' or 'doubles'."
            )

        if self.config.boundary_tolerance_meters < 0:
            raise BounceCourtAnalysisError(
                "boundary_tolerance_meters cannot be negative."
            )

        if not (
            0.0
            <= self.config.minimum_bounce_score
            <= 1.0
        ):
            raise BounceCourtAnalysisError(
                "minimum_bounce_score must be within [0, 1]."
            )

    # ============================================================
    # BOUNCE VALIDATION
    # ============================================================

    @staticmethod
    def validate_bounce(
        bounce: Dict[str, Any],
    ) -> None:
        required = [
            "frame_number",
            "timestamp_seconds",
            "position",
            "bounce_score",
        ]

        for field in required:
            if field not in bounce:
                raise BounceCourtAnalysisError(
                    f"Bounce candidate missing '{field}'."
                )

        position = bounce[
            "position"
        ]

        if (
            "x" not in position
            or "y" not in position
        ):
            raise BounceCourtAnalysisError(
                "Bounce position must contain x and y."
            )

    # ============================================================
    # PIXEL -> COURT
    # ============================================================

    def map_bounce_to_court(
        self,
        bounce: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert bounce pixel coordinates to meters.
        """

        self.validate_bounce(
            bounce
        )

        pixel_x = float(
            bounce["position"]["x"]
        )

        pixel_y = float(
            bounce["position"]["y"]
        )

        try:
            court_x, court_y = (
                self.homography.image_to_court(
                    (
                        pixel_x,
                        pixel_y,
                    )
                )
            )

        except CourtHomographyError as error:
            raise BounceCourtAnalysisError(
                f"Unable to map bounce to court: {error}"
            ) from error

        return {
            "x_meters": round(
                float(court_x),
                4,
            ),
            "y_meters": round(
                float(court_y),
                4,
            ),
        }

    # ============================================================
    # DIMENSIONS
    # ============================================================

    def get_court_dimensions(
        self,
    ) -> Dict[str, float]:
        dimensions = (
            self.homography.geometry.get_dimensions(
                self.config.court_type
            )
        )

        return {
            "width": float(
                dimensions[
                    "width_meters"
                ]
            ),
            "length": float(
                dimensions[
                    "length_meters"
                ]
            ),
        }

    # ============================================================
    # IN / OUT
    # ============================================================

    def classify_line_call(
        self,
        court_x: float,
        court_y: float,
    ) -> Dict[str, Any]:
        """
        Determine whether the bounce is IN or OUT.

        A small tolerance is used around the boundary.
        """

        dimensions = (
            self.get_court_dimensions()
        )

        width = dimensions[
            "width"
        ]

        length = dimensions[
            "length"
        ]

        tolerance = (
            self.config.boundary_tolerance_meters
        )

        inside_with_tolerance = (
            -tolerance
            <= court_x
            <= width + tolerance
            and
            -tolerance
            <= court_y
            <= length + tolerance
        )

        strictly_inside = (
            0.0
            <= court_x
            <= width
            and
            0.0
            <= court_y
            <= length
        )

        on_or_near_line = (
            inside_with_tolerance
            and not strictly_inside
        )

        if strictly_inside:
            call = "IN"

        elif on_or_near_line:
            call = "IN"

        else:
            call = "OUT"

        return {
            "call": call,
            "strictly_inside": (
                strictly_inside
            ),
            "within_boundary_tolerance": (
                inside_with_tolerance
            ),
            "on_or_near_line": (
                on_or_near_line
            ),
        }

    # ============================================================
    # DISTANCE TO BOUNDARY
    # ============================================================

    def calculate_boundary_distances(
        self,
        court_x: float,
        court_y: float,
    ) -> Dict[str, float]:
        """
        Calculate signed distance from each court boundary.

        Positive = inside direction.
        Negative = outside.
        """

        dimensions = (
            self.get_court_dimensions()
        )

        width = dimensions[
            "width"
        ]

        length = dimensions[
            "length"
        ]

        return {
            "left_sideline_meters": round(
                court_x,
                4,
            ),

            "right_sideline_meters": round(
                width - court_x,
                4,
            ),

            "far_baseline_meters": round(
                court_y,
                4,
            ),

            "near_baseline_meters": round(
                length - court_y,
                4,
            ),
        }

    # ============================================================
    # NEAREST LINE
    # ============================================================

    def find_nearest_boundary(
        self,
        distances: Dict[
            str,
            float,
        ],
    ) -> Dict[str, Any]:
        nearest_name = min(
            distances,
            key=lambda key: abs(
                distances[key]
            ),
        )

        nearest_distance = (
            distances[
                nearest_name
            ]
        )

        return {
            "boundary": (
                nearest_name
            ),

            "signed_distance_meters": round(
                nearest_distance,
                4,
            ),

            "absolute_distance_meters": round(
                abs(
                    nearest_distance
                ),
                4,
            ),
        }

    # ============================================================
    # COURT ZONE
    # ============================================================

    def classify_zone(
        self,
        court_x: float,
        court_y: float,
        line_call: str,
    ) -> str:
        if line_call == "OUT":
            return "out"

        return (
            self.homography.geometry
            .classify_court_zone(
                court_x,
                court_y,
                self.config.court_type,
            )
        )

    # ============================================================
    # COMPLETE BOUNCE ANALYSIS
    # ============================================================

    def analyse_bounce(
        self,
        bounce: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fully analyse one bounce candidate.
        """

        self.validate_bounce(
            bounce
        )

        bounce_score = float(
            bounce[
                "bounce_score"
            ]
        )

        mapped_position = (
            self.map_bounce_to_court(
                bounce
            )
        )

        court_x = float(
            mapped_position[
                "x_meters"
            ]
        )

        court_y = float(
            mapped_position[
                "y_meters"
            ]
        )

        line_call = (
            self.classify_line_call(
                court_x,
                court_y,
            )
        )

        boundary_distances = (
            self.calculate_boundary_distances(
                court_x,
                court_y,
            )
        )

        nearest_boundary = (
            self.find_nearest_boundary(
                boundary_distances
            )
        )

        zone = (
            self.classify_zone(
                court_x,
                court_y,
                line_call["call"],
            )
        )

        accepted_bounce = (
            bounce_score
            >= self.config.minimum_bounce_score
        )

        return {
            "frame_number": int(
                bounce[
                    "frame_number"
                ]
            ),

            "timestamp_seconds": round(
                float(
                    bounce[
                        "timestamp_seconds"
                    ]
                ),
                3,
            ),

            "bounce_score": round(
                bounce_score,
                4,
            ),

            "accepted_bounce": (
                accepted_bounce
            ),

            "pixel_position": {
                "x": round(
                    float(
                        bounce[
                            "position"
                        ]["x"]
                    ),
                    3,
                ),
                "y": round(
                    float(
                        bounce[
                            "position"
                        ]["y"]
                    ),
                    3,
                ),
            },

            "court_position": {
                "x_meters": round(
                    court_x,
                    4,
                ),
                "y_meters": round(
                    court_y,
                    4,
                ),
            },

            "court_type": (
                self.config.court_type
            ),

            "line_call": (
                line_call["call"]
            ),

            "strictly_inside": (
                line_call[
                    "strictly_inside"
                ]
            ),

            "on_or_near_line": (
                line_call[
                    "on_or_near_line"
                ]
            ),

            "zone": zone,

            "nearest_boundary": (
                nearest_boundary
            ),

            "boundary_distances": (
                boundary_distances
            ),
        }

    # ============================================================
    # MULTIPLE BOUNCES
    # ============================================================

    def analyse_bounces(
        self,
        bounces: List[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        results = []

        for bounce in bounces:
            results.append(
                self.analyse_bounce(
                    bounce
                )
            )

        return results

    # ============================================================
    # SUMMARY
    # ============================================================

    @staticmethod
    def summarize(
        bounces: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        accepted = [
            bounce
            for bounce in bounces
            if bounce[
                "accepted_bounce"
            ]
        ]

        in_bounces = [
            bounce
            for bounce in accepted
            if bounce[
                "line_call"
            ] == "IN"
        ]

        out_bounces = [
            bounce
            for bounce in accepted
            if bounce[
                "line_call"
            ] == "OUT"
        ]

        line_bounces = [
            bounce
            for bounce in accepted
            if bounce[
                "on_or_near_line"
            ]
        ]

        return {
            "total_candidates": len(
                bounces
            ),

            "accepted_bounces": len(
                accepted
            ),

            "in_bounces": len(
                in_bounces
            ),

            "out_bounces": len(
                out_bounces
            ),

            "near_line_bounces": len(
                line_bounces
            ),
        }