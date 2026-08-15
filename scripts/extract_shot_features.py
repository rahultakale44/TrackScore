from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.ml import (
    ShotFeatureExtractionError,
    ShotFeatureExtractor,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Convert TrackScore ball trajectory metadata "
            "into rally and shot-level ML features."
        )
    )

    parser.add_argument(
        "trajectory_json",
        type=str,
        help=(
            "Path to ball_tracking or "
            "ball_trajectory JSON file."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=(
            "outputs/ml_features/"
            "shot_features.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=(
            "outputs/ml_features/"
            "shot_features.csv"
        ),
    )

    return parser.parse_args()


def load_trajectory(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    trajectory = payload.get(
        "trajectory"
    )

    if trajectory is None:
        raise ShotFeatureExtractionError(
            "Input JSON does not contain 'trajectory'."
        )

    return trajectory


def save_csv(
    rows,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )

        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                fieldnames
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():
    args = parse_arguments()

    try:
        source_path = Path(
            args.trajectory_json
        )

        trajectory = (
            load_trajectory(
                source_path
            )
        )

        extractor = (
            ShotFeatureExtractor()
        )

        dataset = (
            extractor
            .build_feature_dataset(
                trajectory
            )
        )

        output_json = Path(
            args.output_json
        )

        output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_json.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                dataset,
                file,
                indent=4,
            )

        output_csv = Path(
            args.output_csv
        )

        save_csv(
            dataset[
                "shot_features"
            ],
            output_csv,
        )

        print(
            "\nTrackScore Shot Feature Extraction"
        )

        print("=" * 65)

        print(
            f"Trajectory points: "
            f"{len(trajectory)}"
        )

        print(
            f"Rallies detected: "
            f"{dataset['rally_count']}"
        )

        print(
            f"Shots segmented: "
            f"{dataset['shot_count']}"
        )

        print(
            f"\nJSON: "
            f"{output_json.resolve()}"
        )

        print(
            f"CSV: "
            f"{output_csv.resolve()}"
        )

        print("=" * 65)

        print(
            "Shot-level ML feature extraction successful."
        )

    except (
        FileNotFoundError,
        ShotFeatureExtractionError,
        ValueError,
    ) as error:

        print(
            f"\nFeature extraction failed: "
            f"{error}"
        )

        sys.exit(1)

    except Exception as error:

        print(
            f"\nUnexpected error: "
            f"{error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()