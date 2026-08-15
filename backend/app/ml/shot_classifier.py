from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ShotClassifierError(Exception):
    """Raised when shot classification fails."""


@dataclass
class ShotClassifierConfig:
    """
    Configuration for TrackScore shot classification.
    """

    label_column: str = "shot_type"

    test_size: float = 0.20

    random_state: int = 42

    minimum_samples: int = 10

    minimum_classes: int = 2

    model_output_path: str = (
        "models/shot/shot_classifier.joblib"
    )


class ShotClassifier:
    """
    Baseline machine-learning pipeline for tennis-shot classification.

    Supported baseline models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

    The model consumes engineered numerical shot features and
    predicts a labelled tennis-shot class.
    """

    DEFAULT_FEATURE_COLUMNS = [
        "duration_seconds",
        "point_count",

        "start_x",
        "start_y",
        "end_x",
        "end_y",

        "delta_x",
        "delta_y",

        "displacement_pixels",
        "trajectory_distance_pixels",

        "average_speed_pixels_per_second",
        "maximum_speed_pixels_per_second",
        "minimum_speed_pixels_per_second",

        "average_direction_degrees",
        "mean_direction_change_degrees",
        "max_direction_change_degrees",

        "predicted_point_ratio",
    ]

    def __init__(
        self,
        config: ShotClassifierConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else ShotClassifierConfig()
        )

        self._validate_config()

        self.feature_columns: List[str] = list(
            self.DEFAULT_FEATURE_COLUMNS
        )

        self.best_model_name: Optional[str] = None

        self.best_pipeline: Optional[
            Pipeline
        ] = None

        self.training_results: Dict[
            str,
            Dict[str, Any],
        ] = {}

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if not config.label_column:
            raise ShotClassifierError(
                "label_column cannot be empty."
            )

        if not (
            0.0
            < config.test_size
            < 1.0
        ):
            raise ShotClassifierError(
                "test_size must be within (0, 1)."
            )

        if config.minimum_samples <= 0:
            raise ShotClassifierError(
                "minimum_samples must be greater than zero."
            )

        if config.minimum_classes < 2:
            raise ShotClassifierError(
                "minimum_classes must be at least 2."
            )

    # ============================================================
    # LOAD DATA
    # ============================================================

    @staticmethod
    def load_csv(
        csv_path: str,
    ) -> pd.DataFrame:
        path = Path(
            csv_path
        )

        if not path.exists():
            raise ShotClassifierError(
                f"Dataset does not exist: {path}"
            )

        try:
            dataframe = pd.read_csv(
                path
            )

        except Exception as error:
            raise ShotClassifierError(
                f"Unable to read dataset: {error}"
            ) from error

        if dataframe.empty:
            raise ShotClassifierError(
                "Dataset is empty."
            )

        return dataframe

    # ============================================================
    # DATASET VALIDATION
    # ============================================================

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        if dataframe is None:
            raise ShotClassifierError(
                "Dataset cannot be None."
            )

        if dataframe.empty:
            raise ShotClassifierError(
                "Dataset cannot be empty."
            )

        if (
            self.config.label_column
            not in dataframe.columns
        ):
            raise ShotClassifierError(
                f"Dataset must contain label column "
                f"'{self.config.label_column}'."
            )

        available_features = [
            column
            for column in self.feature_columns
            if column in dataframe.columns
        ]

        if not available_features:
            raise ShotClassifierError(
                "Dataset does not contain any supported feature columns."
            )

        if (
            len(dataframe)
            < self.config.minimum_samples
        ):
            raise ShotClassifierError(
                f"Dataset requires at least "
                f"{self.config.minimum_samples} samples."
            )

        labels = (
            dataframe[
                self.config.label_column
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        if (
            len(labels)
            < self.config.minimum_classes
        ):
            raise ShotClassifierError(
                f"Dataset requires at least "
                f"{self.config.minimum_classes} shot classes."
            )

    # ============================================================
    # DATA PREPARATION
    # ============================================================

    def prepare_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
    ]:
        self.validate_dataset(
            dataframe
        )

        available_features = [
            column
            for column in self.feature_columns
            if column in dataframe.columns
        ]

        self.feature_columns = (
            available_features
        )

        cleaned = dataframe.dropna(
            subset=[
                self.config.label_column
            ]
        ).copy()

        x = cleaned[
            self.feature_columns
        ].copy()

        y = (
            cleaned[
                self.config.label_column
            ]
            .astype(str)
        )

        return (
            x,
            y,
        )

    # ============================================================
    # PREPROCESSOR
    # ============================================================

    def build_preprocessor(
        self,
    ) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.feature_columns,
                )
            ],
            remainder="drop",
        )

    # ============================================================
    # MODELS
    # ============================================================

    def build_models(
        self,
    ) -> Dict[str, Any]:
        return {
            "logistic_regression": (
                LogisticRegression(
                    max_iter=2000,
                    random_state=(
                        self.config.random_state
                    ),
                )
            ),

            "random_forest": (
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=(
                        self.config.random_state
                    ),
                    n_jobs=-1,
                )
            ),

            "gradient_boosting": (
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=(
                        self.config.random_state
                    ),
                )
            ),
        }

    # ============================================================
    # EVALUATION
    # ============================================================

    @staticmethod
    def evaluate_predictions(
        y_true,
        y_pred,
    ) -> Dict[str, Any]:
        return {
            "accuracy": round(
                float(
                    accuracy_score(
                        y_true,
                        y_pred,
                    )
                ),
                4,
            ),

            "precision_macro": round(
                float(
                    precision_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),

            "recall_macro": round(
                float(
                    recall_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),

            "f1_macro": round(
                float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),

            "classification_report": (
                classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                )
            ),

            "confusion_matrix": (
                confusion_matrix(
                    y_true,
                    y_pred,
                ).tolist()
            ),
        }

    # ============================================================
    # TRAINING
    # ============================================================

    def train(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        x, y = (
            self.prepare_dataset(
                dataframe
            )
        )

        class_counts = (
            y.value_counts()
        )

        use_stratify = (
            class_counts.min()
            >= 2
        )

        stratify = (
            y
            if use_stratify
            else None
        )

        try:
            (
                x_train,
                x_test,
                y_train,
                y_test,
            ) = train_test_split(
                x,
                y,
                test_size=(
                    self.config.test_size
                ),
                random_state=(
                    self.config.random_state
                ),
                stratify=stratify,
            )

        except ValueError as error:
            raise ShotClassifierError(
                f"Unable to split dataset: {error}"
            ) from error

        models = self.build_models()

        results: Dict[
            str,
            Dict[str, Any],
        ] = {}

        best_f1 = -1.0

        best_model_name = None

        best_pipeline = None

        for (
            model_name,
            estimator,
        ) in models.items():

            preprocessor = (
                self.build_preprocessor()
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "classifier",
                        estimator,
                    ),
                ]
            )

            try:
                pipeline.fit(
                    x_train,
                    y_train,
                )

                predictions = (
                    pipeline.predict(
                        x_test
                    )
                )

            except Exception as error:
                results[
                    model_name
                ] = {
                    "status": "failed",
                    "error": str(
                        error
                    ),
                }

                continue

            metrics = (
                self.evaluate_predictions(
                    y_test,
                    predictions,
                )
            )

            results[
                model_name
            ] = {
                "status": "success",
                "metrics": metrics,
            }

            current_f1 = float(
                metrics[
                    "f1_macro"
                ]
            )

            if current_f1 > best_f1:
                best_f1 = (
                    current_f1
                )

                best_model_name = (
                    model_name
                )

                best_pipeline = (
                    pipeline
                )

        if best_pipeline is None:
            raise ShotClassifierError(
                "All candidate models failed to train."
            )

        self.best_model_name = (
            best_model_name
        )

        self.best_pipeline = (
            best_pipeline
        )

        self.training_results = (
            results
        )

        return {
            "samples": len(
                dataframe
            ),

            "training_samples": len(
                x_train
            ),

            "test_samples": len(
                x_test
            ),

            "classes": sorted(
                y.unique().tolist()
            ),

            "feature_columns": (
                self.feature_columns
            ),

            "model_results": (
                results
            ),

            "best_model": (
                self.best_model_name
            ),

            "best_f1_macro": round(
                best_f1,
                4,
            ),
        }

    # ============================================================
    # SAVE MODEL
    # ============================================================

    def save_model(
        self,
        output_path: Optional[
            str
        ] = None,
    ) -> Path:
        if self.best_pipeline is None:
            raise ShotClassifierError(
                "No trained model is available."
            )

        target = Path(
            output_path
            or self.config.model_output_path
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact = {
            "model": (
                self.best_pipeline
            ),

            "model_name": (
                self.best_model_name
            ),

            "feature_columns": (
                self.feature_columns
            ),

            "label_column": (
                self.config.label_column
            ),
        }

        joblib.dump(
            artifact,
            target,
        )

        return target

    # ============================================================
    # LOAD MODEL
    # ============================================================

    @staticmethod
    def load_model(
        model_path: str,
    ) -> Dict[str, Any]:
        path = Path(
            model_path
        )

        if not path.exists():
            raise ShotClassifierError(
                f"Model does not exist: {path}"
            )

        try:
            artifact = joblib.load(
                path
            )

        except Exception as error:
            raise ShotClassifierError(
                f"Unable to load model: {error}"
            ) from error

        required = [
            "model",
            "model_name",
            "feature_columns",
        ]

        for field in required:
            if field not in artifact:
                raise ShotClassifierError(
                    f"Invalid model artifact: missing '{field}'."
                )

        return artifact

    # ============================================================
    # PREDICTION
    # ============================================================

    @staticmethod
    def predict(
        model_artifact: Dict[
            str,
            Any,
        ],
        features: Dict[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        model = model_artifact[
            "model"
        ]

        feature_columns = (
            model_artifact[
                "feature_columns"
            ]
        )

        row = {
            column: features.get(
                column,
                np.nan,
            )
            for column in feature_columns
        }

        dataframe = pd.DataFrame(
            [
                row
            ]
        )

        predicted_class = (
            model.predict(
                dataframe
            )[0]
        )

        result = {
            "shot_type": str(
                predicted_class
            )
        }

        if hasattr(
            model,
            "predict_proba",
        ):
            probabilities = (
                model.predict_proba(
                    dataframe
                )[0]
            )

            classes = (
                model.classes_
            )

            probability_map = {
                str(label): round(
                    float(probability),
                    4,
                )
                for (
                    label,
                    probability,
                ) in zip(
                    classes,
                    probabilities,
                )
            }

            result[
                "probabilities"
            ] = probability_map

            result[
                "confidence"
            ] = round(
                max(
                    probability_map.values()
                ),
                4,
            )

        return result