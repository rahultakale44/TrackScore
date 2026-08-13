import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.vision import (
    CourtHomography,
)


def main():
    """
    Demonstrate homography using synthetic image court corners.

    These pixel coordinates are only demonstration data.

    Later an ML court-keypoint detector will supply these
    coordinates automatically from actual tennis videos.
    """

    image_corners = [
        [750, 250],
        [1170, 250],
        [300, 950],
        [1620, 950],
    ]

    homography = CourtHomography()

    calibration = homography.calibrate(
        image_corners,
        court_type="singles",
    )

    center_pixel = (
        960,
        600,
    )

    center_analysis = (
        homography.analyse_image_point(
            center_pixel
        )
    )

    real_center = (
        8.23 / 2,
        23.77 / 2,
    )

    projected_center = (
        homography.court_to_image(
            real_center
        )
    )

    print(
        "\nTrackScore Court Homography"
    )

    print("=" * 60)

    print(
        f"Court Type: "
        f"{calibration.court_type}"
    )

    print(
        f"Court Size: "
        f"{calibration.court_width_meters} m "
        f"x "
        f"{calibration.court_length_meters} m"
    )

    print(
        f"Reprojection Error: "
        f"{calibration.reprojection_error}"
    )

    print(
        "\nImage Point Analysis:"
    )

    print(
        json.dumps(
            center_analysis,
            indent=4,
        )
    )

    print(
        "\nReal Court Center:"
    )

    print(
        real_center
    )

    print(
        "\nProjected Court Center Pixel:"
    )

    print(
        projected_center
    )

    print(
        "\nHomography Matrix:"
    )

    print(
        calibration.matrix
    )

    print("=" * 60)

    print(
        "Homography calibration successful."
    )


if __name__ == "__main__":
    main()