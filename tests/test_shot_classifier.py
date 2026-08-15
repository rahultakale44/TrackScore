import pandas as pd
import pytest

from backend.app.ml.shot_classifier import (
    ShotClassifier,
    ShotClassifierConfig,
    ShotClassifierError,
)


def create_dataset():
    rows = []

    for index in range(
        20
    ):
        rows.append(
            {
                "duration_seconds": (
                    0.5
                    + index * 0.01
                ),
                "point_count": 5,
                "start_x": 100,
                "start_y": 200,
                "end_x": 400,
                "end_y": 300,
                "delta_x": 300,
                "delta_y": 100,
                "displacement_pixels": 316,
                "trajectory_distance_pixels": 330,
                "average_speed_pixels_per_second": 600,
                "maximum_speed_pixels_per_second": 700,
                "minimum_speed_pixels_per_second": 450,
                "average_direction_degrees": 20,
                "mean_direction_change_degrees": 15,
                "max_direction_change_degrees": 40,
                "predicted_point_ratio": 0.05,
                "shot_type": "forehand",
            }
        )

    for index in range(
        20
    ):
        rows.append(
            {
                "duration_seconds": (
                    0.9
                    + index * 0.01
                ),
                "point_count": 6,
                "start_x": 300,
                "start_y": 300,
                "end_x": 200,
                "end_y": 500,
                "delta_x": -100,
                "delta_y": 200,
                "displacement_pixels": 223,
                "trajectory_distance_pixels": 240,
                "average_speed_pixels_per_second": 300,
                "maximum_speed_pixels_per_second": 350,
                "minimum_speed_pixels_per_second": 220,
                "average_direction_degrees": 100,
                "mean_direction_change_degrees": 30,
                "max_direction_change_degrees": 70,
                "predicted_point_ratio": 0.08,
                "shot_type": "drop",
            }
        )

    return pd.DataFrame(
        rows
    )


def test_invalid_test_size():
    config = (
        ShotClassifierConfig(
            test_size=1.0
        )
    )

    with pytest.raises(
        ShotClassifierError
    ):
        ShotClassifier(
            config
        )


def test_missing_label():
    classifier = (
        ShotClassifier()
    )

    dataframe = (
        create_dataset()
        .drop(
            columns=[
                "shot_type"
            ]
        )
    )

    with pytest.raises(
        ShotClassifierError
    ):
        classifier.validate_dataset(
            dataframe
        )


def test_prepare_dataset():
    classifier = (
        ShotClassifier()
    )

    dataframe = (
        create_dataset()
    )

    x, y = (
        classifier.prepare_dataset(
            dataframe
        )
    )

    assert len(
        x
    ) == 40

    assert len(
        y
    ) == 40


def test_build_models():
    classifier = (
        ShotClassifier()
    )

    models = (
        classifier.build_models()
    )

    assert (
        "logistic_regression"
        in models
    )

    assert (
        "random_forest"
        in models
    )

    assert (
        "gradient_boosting"
        in models
    )


def test_training():
    classifier = (
        ShotClassifier()
    )

    results = (
        classifier.train(
            create_dataset()
        )
    )

    assert (
        results[
            "best_model"
        ]
        is not None
    )

    assert (
        classifier.best_pipeline
        is not None
    )


def test_save_and_load(
    tmp_path,
):
    classifier = (
        ShotClassifier()
    )

    classifier.train(
        create_dataset()
    )

    model_path = (
        tmp_path
        / "model.joblib"
    )

    classifier.save_model(
        str(
            model_path
        )
    )

    artifact = (
        ShotClassifier
        .load_model(
            str(
                model_path
            )
        )
    )

    assert (
        "model"
        in artifact
    )


def test_prediction(
    tmp_path,
):
    classifier = (
        ShotClassifier()
    )

    dataframe = (
        create_dataset()
    )

    classifier.train(
        dataframe
    )

    model_path = (
        tmp_path
        / "model.joblib"
    )

    classifier.save_model(
        str(
            model_path
        )
    )

    artifact = (
        ShotClassifier
        .load_model(
            str(
                model_path
            )
        )
    )

    sample = (
        dataframe
        .iloc[0]
        .to_dict()
    )

    result = (
        ShotClassifier
        .predict(
            artifact,
            sample,
        )
    )

    assert (
        "shot_type"
        in result
    )