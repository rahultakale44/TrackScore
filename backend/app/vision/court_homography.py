from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .court_geometry import CourtGeometry


class CourtHomographyError(Exception):
    """Raised when court homography calibration or transformation fails."""


@dataclass
class HomographyCalibration:
    """
    Stores the result of a court homography calibration.
    """

    matrix: np.ndarray
    inverse_matrix: np.ndarray
    court_type: str
    court_width_meters: float
    court_length_meters: float
    reprojection_error: float


class CourtHomography:
    """
    Maps tennis court coordinates between:

        Image pixel space
                ↕
        Real-world court space in meters

    Expected image-point order:

        1. far_left
        2. far_right
        3. near_left
        4. near_right

    Example:

        image:
            (810, 310)
            (1120, 310)
            (350, 950)
            (1580, 950)

        court:
            (0.0, 0.0)
            (8.23, 0.0)
            (0.0, 23.77)
            (8.23, 23.77)
    """

    REQUIRED_POINT_COUNT = 4

    def __init__(
        self,
        geometry: CourtGeometry | None = None,
    ):
        self.geometry = (
            geometry
            if geometry is not None
            else CourtGeometry()
        )

        self.calibration: HomographyCalibration | None = None

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_points(
        points: Sequence[Sequence[float]],
        name: str,
        minimum_points: int = 4,
    ) -> np.ndarray:
        """
        Validate and convert 2D coordinates to float32 array.
        """

        array = np.asarray(
            points,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise CourtHomographyError(
                f"{name} must be a 2D array."
            )

        if array.shape[1] != 2:
            raise CourtHomographyError(
                f"{name} must contain x and y coordinates."
            )

        if array.shape[0] < minimum_points:
            raise CourtHomographyError(
                f"{name} requires at least {minimum_points} points."
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise CourtHomographyError(
                f"{name} contains invalid numeric values."
            )

        return array

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
    ) -> None:
        if matrix is None:
            raise CourtHomographyError(
                "Homography matrix cannot be None."
            )

        if not isinstance(
            matrix,
            np.ndarray,
        ):
            raise CourtHomographyError(
                "Homography matrix must be a NumPy array."
            )

        if matrix.shape != (3, 3):
            raise CourtHomographyError(
                "Homography matrix must have shape (3, 3)."
            )

        if not np.all(
            np.isfinite(matrix)
        ):
            raise CourtHomographyError(
                "Homography matrix contains invalid values."
            )

    # ============================================================
    # STANDARD REAL-WORLD COURT CORNERS
    # ============================================================

    def get_real_court_corners(
        self,
        court_type: str = "singles",
    ) -> np.ndarray:
        """
        Generate standard court corner coordinates in meters.

        Returned order:

            far-left
            far-right
            near-left
            near-right
        """

        dimensions = self.geometry.get_dimensions(
            court_type
        )

        width = float(
            dimensions["width_meters"]
        )

        length = float(
            dimensions["length_meters"]
        )

        return np.array(
            [
                [0.0, 0.0],
                [width, 0.0],
                [0.0, length],
                [width, length],
            ],
            dtype=np.float32,
        )

    # ============================================================
    # CALIBRATION
    # ============================================================

    def calibrate(
        self,
        image_points: Sequence[
            Sequence[float]
        ],
        court_type: str = "singles",
    ) -> HomographyCalibration:
        """
        Calibrate using exactly four visible court corners.

        image_points order:

            far-left
            far-right
            near-left
            near-right
        """

        image_array = self._validate_points(
            image_points,
            "image_points",
            minimum_points=4,
        )

        if image_array.shape[0] != 4:
            raise CourtHomographyError(
                "Four-point calibration requires exactly 4 image points."
            )

        real_array = self.get_real_court_corners(
            court_type
        )

        matrix = cv2.getPerspectiveTransform(
            image_array,
            real_array,
        )

        self._validate_matrix(
            matrix
        )

        try:
            inverse_matrix = np.linalg.inv(
                matrix
            )

        except np.linalg.LinAlgError as error:
            raise CourtHomographyError(
                "Homography matrix is singular and cannot be inverted."
            ) from error

        dimensions = self.geometry.get_dimensions(
            court_type
        )

        calibration = HomographyCalibration(
            matrix=matrix,
            inverse_matrix=inverse_matrix,
            court_type=court_type,
            court_width_meters=float(
                dimensions["width_meters"]
            ),
            court_length_meters=float(
                dimensions["length_meters"]
            ),
            reprojection_error=self._calculate_reprojection_error(
                image_array,
                real_array,
                matrix,
            ),
        )

        self.calibration = calibration

        return calibration

    def calibrate_from_correspondences(
        self,
        image_points: Sequence[
            Sequence[float]
        ],
        real_points: Sequence[
            Sequence[float]
        ],
        court_type: str = "singles",
    ) -> HomographyCalibration:
        """
        Estimate homography using 4 or more image/real-world
        point correspondences.

        This will later be useful when a court-keypoint ML model
        predicts several court landmarks.
        """

        image_array = self._validate_points(
            image_points,
            "image_points",
            minimum_points=4,
        )

        real_array = self._validate_points(
            real_points,
            "real_points",
            minimum_points=4,
        )

        if (
            image_array.shape[0]
            != real_array.shape[0]
        ):
            raise CourtHomographyError(
                "Image and real-world point counts must match."
            )

        matrix, inlier_mask = cv2.findHomography(
            image_array,
            real_array,
            method=cv2.RANSAC,
        )

        if matrix is None:
            raise CourtHomographyError(
                "OpenCV could not estimate a valid homography."
            )

        self._validate_matrix(
            matrix
        )

        try:
            inverse_matrix = np.linalg.inv(
                matrix
            )

        except np.linalg.LinAlgError as error:
            raise CourtHomographyError(
                "Estimated homography matrix is singular."
            ) from error

        dimensions = self.geometry.get_dimensions(
            court_type
        )

        calibration = HomographyCalibration(
            matrix=matrix,
            inverse_matrix=inverse_matrix,
            court_type=court_type,
            court_width_meters=float(
                dimensions["width_meters"]
            ),
            court_length_meters=float(
                dimensions["length_meters"]
            ),
            reprojection_error=self._calculate_reprojection_error(
                image_array,
                real_array,
                matrix,
            ),
        )

        self.calibration = calibration

        return calibration

    # ============================================================
    # REPROJECTION ERROR
    # ============================================================

    @staticmethod
    def _calculate_reprojection_error(
        source_points: np.ndarray,
        target_points: np.ndarray,
        matrix: np.ndarray,
    ) -> float:
        """
        Measure average mapping error.
        """

        transformed = cv2.perspectiveTransform(
            source_points.reshape(
                -1,
                1,
                2,
            ),
            matrix,
        ).reshape(
            -1,
            2,
        )

        distances = np.linalg.norm(
            transformed - target_points,
            axis=1,
        )

        return round(
            float(
                np.mean(
                    distances
                )
            ),
            6,
        )

    # ============================================================
    # IMAGE → COURT
    # ============================================================

    def image_to_court(
        self,
        point: Sequence[float],
    ) -> Tuple[float, float]:
        """
        Convert one image pixel point to court meters.
        """

        calibration = self._require_calibration()

        array = self._validate_points(
            [point],
            "point",
            minimum_points=1,
        )

        transformed = cv2.perspectiveTransform(
            array.reshape(
                -1,
                1,
                2,
            ),
            calibration.matrix,
        )

        x, y = transformed[
            0,
            0,
        ]

        return (
            round(
                float(x),
                4,
            ),
            round(
                float(y),
                4,
            ),
        )

    def image_points_to_court(
        self,
        points: Sequence[
            Sequence[float]
        ],
    ) -> List[
        Tuple[float, float]
    ]:
        """
        Convert multiple image pixel coordinates to meters.
        """

        calibration = self._require_calibration()

        array = self._validate_points(
            points,
            "points",
            minimum_points=1,
        )

        transformed = cv2.perspectiveTransform(
            array.reshape(
                -1,
                1,
                2,
            ),
            calibration.matrix,
        ).reshape(
            -1,
            2,
        )

        return [
            (
                round(
                    float(point[0]),
                    4,
                ),
                round(
                    float(point[1]),
                    4,
                ),
            )
            for point in transformed
        ]

    # ============================================================
    # COURT → IMAGE
    # ============================================================

    def court_to_image(
        self,
        point: Sequence[float],
    ) -> Tuple[float, float]:
        """
        Convert real court coordinate in meters into image pixels.
        """

        calibration = self._require_calibration()

        array = self._validate_points(
            [point],
            "point",
            minimum_points=1,
        )

        transformed = cv2.perspectiveTransform(
            array.reshape(
                -1,
                1,
                2,
            ),
            calibration.inverse_matrix,
        )

        x, y = transformed[
            0,
            0,
        ]

        return (
            round(
                float(x),
                4,
            ),
            round(
                float(y),
                4,
            ),
        )

    def court_points_to_image(
        self,
        points: Sequence[
            Sequence[float]
        ],
    ) -> List[
        Tuple[float, float]
    ]:
        """
        Convert several real-world court coordinates to pixels.
        """

        calibration = self._require_calibration()

        array = self._validate_points(
            points,
            "points",
            minimum_points=1,
        )

        transformed = cv2.perspectiveTransform(
            array.reshape(
                -1,
                1,
                2,
            ),
            calibration.inverse_matrix,
        ).reshape(
            -1,
            2,
        )

        return [
            (
                round(
                    float(point[0]),
                    4,
                ),
                round(
                    float(point[1]),
                    4,
                ),
            )
            for point in transformed
        ]

    # ============================================================
    # COURT POSITION INFORMATION
    # ============================================================

    def analyse_image_point(
        self,
        point: Sequence[float],
    ) -> Dict[str, object]:
        """
        Convert an image point and describe its tennis-court state.
        """

        calibration = self._require_calibration()

        x, y = self.image_to_court(
            point
        )

        is_inside = (
            0.0
            <= x
            <= calibration.court_width_meters
            and
            0.0
            <= y
            <= calibration.court_length_meters
        )

        if is_inside:
            zone = (
                self.geometry.classify_court_zone(
                    x,
                    y,
                    calibration.court_type,
                )
            )

        else:
            zone = "out"

        return {
            "image_x": float(
                point[0]
            ),
            "image_y": float(
                point[1]
            ),
            "court_x_meters": x,
            "court_y_meters": y,
            "court_type": (
                calibration.court_type
            ),
            "inside_court": is_inside,
            "zone": zone,
        }

    # ============================================================
    # DISTANCE
    # ============================================================

    @staticmethod
    def calculate_real_distance(
        point_a: Sequence[float],
        point_b: Sequence[float],
    ) -> float:
        """
        Calculate Euclidean distance between two court
        coordinates in meters.
        """

        if len(point_a) != 2:
            raise CourtHomographyError(
                "point_a must contain x and y."
            )

        if len(point_b) != 2:
            raise CourtHomographyError(
                "point_b must contain x and y."
            )

        dx = (
            float(point_b[0])
            - float(point_a[0])
        )

        dy = (
            float(point_b[1])
            - float(point_a[1])
        )

        distance = np.hypot(
            dx,
            dy,
        )

        return round(
            float(distance),
            4,
        )

    # ============================================================
    # CALIBRATION STATE
    # ============================================================

    def _require_calibration(
        self,
    ) -> HomographyCalibration:
        if self.calibration is None:
            raise CourtHomographyError(
                "Court homography has not been calibrated."
            )

        return self.calibration

    def is_calibrated(
        self,
    ) -> bool:
        return (
            self.calibration
            is not None
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_calibration_summary(
        self,
    ) -> Dict[str, object]:
        calibration = self._require_calibration()

        return {
            "calibrated": True,
            "court_type": (
                calibration.court_type
            ),
            "court_width_meters": (
                calibration.court_width_meters
            ),
            "court_length_meters": (
                calibration.court_length_meters
            ),
            "reprojection_error": (
                calibration.reprojection_error
            ),
            "homography_matrix": (
                calibration.matrix.tolist()
            ),
            "inverse_homography_matrix": (
                calibration.inverse_matrix.tolist()
            ),
        }