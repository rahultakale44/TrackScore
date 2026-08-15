from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/processed/"
            "demo_shot_dataset.csv"
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
    )

    return parser.parse_args()


def generate_class(
    rng,
    label,
    samples,
    speed_mean,
    delta_y_mean,
    direction_mean,
    duration_mean,
):
    rows = []

    for _ in range(
        samples
    ):
        start_x = rng.uniform(
            200,
            1700,
        )

        start_y = rng.uniform(
            200,
            850,
        )

        delta_x = rng.normal(
            direction_mean,
            100,
        )

        delta_y = rng.normal(
            delta_y_mean,
            60,
        )

        end_x = (
            start_x
            + delta_x
        )

        end_y = (
            start_y
            + delta_y
        )

        duration = max(
            0.1,
            rng.normal(
                duration_mean,
                0.1,
            ),
        )

        average_speed = max(
            1.0,
            rng.normal(
                speed_mean,
                50,
            ),
        )

        displacement = float(
            np.hypot(
                delta_x,
                delta_y,
            )
        )

        rows.append(
            {
                "duration_seconds": duration,
                "point_count": int(
                    rng.integers(
                        3,
                        12,
                    )
                ),

                "start_x": start_x,
                "start_y": start_y,

                "end_x": end_x,
                "end_y": end_y,

                "delta_x": delta_x,
                "delta_y": delta_y,

                "displacement_pixels": (
                    displacement
                ),

                "trajectory_distance_pixels": (
                    displacement
                    * rng.uniform(
                        1.0,
                        1.25,
                    )
                ),

                "average_speed_pixels_per_second": (
                    average_speed
                ),

                "maximum_speed_pixels_per_second": (
                    average_speed
                    * rng.uniform(
                        1.05,
                        1.4,
                    )
                ),

                "minimum_speed_pixels_per_second": (
                    average_speed
                    * rng.uniform(
                        0.5,
                        0.9,
                    )
                ),

                "average_direction_degrees": (
                    rng.uniform(
                        -180,
                        180,
                    )
                ),

                "mean_direction_change_degrees": (
                    rng.uniform(
                        5,
                        45,
                    )
                ),

                "max_direction_change_degrees": (
                    rng.uniform(
                        20,
                        100,
                    )
                ),

                "predicted_point_ratio": (
                    rng.uniform(
                        0,
                        0.2,
                    )
                ),

                "shot_type": label,
            }
        )

    return rows


def main():
    args = parse_arguments()

    rng = np.random.default_rng(
        42
    )

    rows = []

    rows.extend(
        generate_class(
            rng=rng,
            label="forehand",
            samples=(
                args.samples_per_class
            ),
            speed_mean=650,
            delta_y_mean=-80,
            direction_mean=250,
            duration_mean=0.65,
        )
    )

    rows.extend(
        generate_class(
            rng=rng,
            label="backhand",
            samples=(
                args.samples_per_class
            ),
            speed_mean=570,
            delta_y_mean=-40,
            direction_mean=-220,
            duration_mean=0.72,
        )
    )

    rows.extend(
        generate_class(
            rng=rng,
            label="serve",
            samples=(
                args.samples_per_class
            ),
            speed_mean=950,
            delta_y_mean=320,
            direction_mean=50,
            duration_mean=0.45,
        )
    )

    rows.extend(
        generate_class(
            rng=rng,
            label="drop",
            samples=(
                args.samples_per_class
            ),
            speed_mean=300,
            delta_y_mean=100,
            direction_mean=30,
            duration_mean=0.95,
        )
    )

    dataframe = pd.DataFrame(
        rows
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output,
        index=False,
    )

    print(
        f"Created demo dataset: {output.resolve()}"
    )

    print(
        f"Samples: {len(dataframe)}"
    )

    print(
        dataframe[
            "shot_type"
        ].value_counts()
    )


if __name__ == "__main__":
    main()