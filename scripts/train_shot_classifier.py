from __future__ import annotations

import argparse
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
    ShotClassifier,
    ShotClassifierConfig,
    ShotClassifierError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train TrackScore baseline "
            "tennis-shot classifiers."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
    )

    parser.add_argument(
        "--label-column",
        type=str,
        default="shot_type",
    )

    parser.add_argument(
        "--model-output",
        type=str,
        default=(
            "models/shot/"
            "shot_classifier.joblib"
        ),
    )

    parser.add_argument(
        "--metrics-output",
        type=str,
        default=(
            "outputs/ml_training/"
            "shot_classifier_metrics.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        classifier = ShotClassifier(
            ShotClassifierConfig(
                label_column=(
                    args.label_column
                ),
                model_output_path=(
                    args.model_output
                ),
            )
        )

        dataframe = (
            classifier.load_csv(
                args.dataset
            )
        )

        results = (
            classifier.train(
                dataframe
            )
        )

        model_path = (
            classifier.save_model()
        )

        metrics_path = Path(
            args.metrics_output
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=4,
            )

        print(
            "\nTrackScore Shot Classification"
        )

        print("=" * 65)

        print(
            f"Samples: "
            f"{results['samples']}"
        )

        print(
            f"Classes: "
            f"{results['classes']}"
        )

        print(
            "\nModel comparison:"
        )

        for (
            model_name,
            result,
        ) in results[
            "model_results"
        ].items():

            if (
                result["status"]
                == "success"
            ):
                metrics = (
                    result[
                        "metrics"
                    ]
                )

                print(
                    (
                        f"{model_name}: "
                        f"Accuracy="
                        f"{metrics['accuracy']:.3f}, "
                        f"F1="
                        f"{metrics['f1_macro']:.3f}"
                    )
                )

            else:
                print(
                    (
                        f"{model_name}: "
                        f"FAILED"
                    )
                )

        print(
            f"\nBest Model: "
            f"{results['best_model']}"
        )

        print(
            f"Best F1: "
            f"{results['best_f1_macro']}"
        )

        print(
            f"\nModel saved: "
            f"{model_path.resolve()}"
        )

        print(
            f"Metrics saved: "
            f"{metrics_path.resolve()}"
        )

        print("=" * 65)

    except ShotClassifierError as error:

        print(
            f"\nTraining failed: {error}"
        )

        sys.exit(1)

    except Exception as error:

        print(
            f"\nUnexpected error: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()