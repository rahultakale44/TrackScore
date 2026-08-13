import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.vision import CourtGeometry


def main():
    geometry = CourtGeometry()

    summary = geometry.get_summary()

    singles = geometry.get_dimensions(
        "singles"
    )

    doubles = geometry.get_dimensions(
        "doubles"
    )

    print(
        "\nTrackScore Tennis Court Geometry"
    )

    print("=" * 55)

    print(
        f"Court Length : "
        f"{summary['court_length_meters']} m"
    )

    print(
        f"Singles Width: "
        f"{singles['width_meters']} m"
    )

    print(
        f"Doubles Width: "
        f"{doubles['width_meters']} m"
    )

    print(
        f"Net → Baseline: "
        f"{summary['net_to_baseline_meters']} m"
    )

    print(
        f"Service Line → Net: "
        f"{summary['service_line_distance_from_net_meters']} m"
    )

    print(
        "\nCourt Keypoints:"
    )

    print(
        json.dumps(
            summary["keypoints"],
            indent=4,
        )
    )

    print("=" * 55)

    print(
        "Court geometry initialized successfully."
    )


if __name__ == "__main__":
    main()