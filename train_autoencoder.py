"""Train and evaluate a normal-only autoencoder for network anomaly detection."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_CANDIDATES = ("data/bin_data.csv", "bin_data.csv")
LABEL_COLUMN = "label"
MODEL_DIRECTORY = Path("models")
DEFAULT_SEED = 42
NORMAL_LABEL_CANDIDATES = ("normal", "benign", "legitimate", "background", "0", "false")


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV and standardise the name of its label column."""
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("The dataset has no records.")
    matching = [column for column in data.columns if str(column).strip().lower() == LABEL_COLUMN]
    if len(matching) != 1:
        raise ValueError("Expected exactly one label column named 'label' (case-insensitive).")
    if matching[0] != LABEL_COLUMN:
        data = data.rename(columns={matching[0]: LABEL_COLUMN})
    if data.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")
    if data[LABEL_COLUMN].isna().any():
        raise ValueError("The label column contains missing values; resolve them before training.")
    return data.replace([np.inf, -np.inf], np.nan)


def _normalise_label(value: object) -> str:
    """Return a stable string representation for comparing label values."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def map_labels(labels: pd.Series, normal_label: str | None = None) -> tuple[np.ndarray, str]:
    """Map normal records to 0 and every other labelled class to 1."""
    normalised = labels.map(_normalise_label)
    observed = set(normalised.unique())

    if normal_label is not None:
        normal_token = _normalise_label(normal_label)
        if normal_token not in observed:
            raise ValueError(
                f"Normal label {normal_label!r} was not found. "
                f"Observed labels: {sorted(observed)}"
            )
    else:
        normal_token = next(
            (candidate for candidate in NORMAL_LABEL_CANDIDATES if candidate in observed),
            None,
        )
        if normal_token is None:
            raise ValueError(
                "Could not safely infer the normal class from labels "
                f"{sorted(observed)}. Pass --normal-label with the exact normal label."
            )

    binary = (normalised != normal_token).astype(np.int8).to_numpy()
    if len(np.unique(binary)) != 2:
        raise ValueError(
            "The dataset must contain at least one normal record and one non-normal record."
        )
    return binary, normal_token


def split_indices(
    binary_labels: np.ndarray, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create disjoint 60% development, 20% calibration, and 20% test splits."""
    indices = np.arange(len(binary_labels))
    _, counts = np.unique(binary_labels, return_counts=True)
    if len(counts) != 2 or int(counts.min()) < 5:
        raise ValueError(
            "Need at least five normal and five abnormal records to create stratified splits."
        )

    development, test = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=binary_labels,
    )
    training, calibration = train_test_split(
        development,
        test_size=0.25,
        random_state=seed,
        stratify=binary_labels[development],
    )
    if len(set(training) & set(calibration) or set()) or len(set(training) & set(test)):
        raise RuntimeError("Internal split error: partitions overlap.")
    if len(set(calibration) & set(test)):
        raise RuntimeError("Internal split error: calibration and test overlap.")
    return np.asarray(training), np.asarray(calibration), np.asarray(test)


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessor that handles numeric and categorical telemetry."""
    numeric_columns = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=32,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_autoencoder(input_dimension: int):
    """Build an autoencoder with a linear output for scaled numeric features."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dimension,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(input_dimension, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_errors(model, matrix: np.ndarray) -> np.ndarray:
    """Calculate per-record mean squared reconstruction error."""
    reconstructed = model.predict(matrix, verbose=0)
    return np.mean(np.square(matrix - reconstructed), axis=1)


def calculate_metrics(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    """Calculate held-out binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, object] = {
        "test_records": int(len(labels)),
        "normal_records": int(np.sum(labels == 0)),
        "abnormal_records": int(np.sum(labels == 1)),
        "thresholded_anomalies": int(np.sum(predictions == 1)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def save_artifacts(
    model,
    preprocessor: ColumnTransformer,
    feature_columns: list[str],
    threshold: float,
    threshold_quantile: float,
    calibration_count: int,
    normal_label: str,
    metrics: dict[str, object],
    seed: int,
) -> None:
    """Save model, fitted preprocessing, threshold metadata and test metrics."""
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIRECTORY / "autoencoder.keras")
    with (MODEL_DIRECTORY / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle)
    with (MODEL_DIRECTORY / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, indent=2)
    with (MODEL_DIRECTORY / "threshold.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": float(threshold),
                "calibration_quantile": float(threshold_quantile),
                "calibration_normal_records": int(calibration_count),
                "normal_label": normal_label,
                "seed": int(seed),
                "method": "quantile of normal-only calibration reconstruction errors",
            },
            handle,
            indent=2,
        )
    with (MODEL_DIRECTORY / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Artifacts and held-out metrics saved to {MODEL_DIRECTORY}/.")


def train_pipeline(
    data_path: str | Path,
    normal_label: str | None = None,
    threshold_quantile: float = 99.0,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Train on normal development data, calibrate on normal data, and test once."""
    if not 90.0 <= threshold_quantile < 100.0:
        raise ValueError("--threshold-quantile must be at least 90 and less than 100.")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    data = load_data(data_path)
    binary_labels, normal_token = map_labels(data[LABEL_COLUMN], normal_label)
    feature_columns = [column for column in data.columns if column != LABEL_COLUMN]
    if not feature_columns:
        raise ValueError("No feature columns remain after removing the label.")

    raw_features = data[feature_columns].copy()
    training_indices, calibration_indices, test_indices = split_indices(binary_labels, seed)

    normal_training_indices = training_indices[binary_labels[training_indices] == 0]
    normal_calibration_indices = calibration_indices[binary_labels[calibration_indices] == 0]
    if len(normal_training_indices) < 20:
        raise ValueError(
            "Need at least 20 normal records in the training partition for a stable fit; "
            f"found {len(normal_training_indices)}."
        )
    if len(normal_calibration_indices) < 5:
        raise ValueError(
            "Need at least five normal records in the calibration partition; "
            f"found {len(normal_calibration_indices)}."
        )

    # Drop columns with no observations in normal training data, before fitting anything.
    available_columns = [
        column
        for column in feature_columns
        if raw_features.iloc[normal_training_indices][column].notna().any()
    ]
    if not available_columns:
        raise ValueError("All feature columns are empty among normal training records.")
    raw_features = raw_features[available_columns]

    normal_train = raw_features.iloc[normal_training_indices]
    calibration_normal = raw_features.iloc[normal_calibration_indices]
    test_features = raw_features.iloc[test_indices]
    test_labels = binary_labels[test_indices]

    preprocessor = make_preprocessor(normal_train)
    x_train = np.asarray(preprocessor.fit_transform(normal_train), dtype=np.float32)
    x_calibration = np.asarray(
        preprocessor.transform(calibration_normal), dtype=np.float32
    )
    x_test = np.asarray(preprocessor.transform(test_features), dtype=np.float32)

    if not np.isfinite(x_train).all() or not np.isfinite(x_calibration).all() or not np.isfinite(x_test).all():
        raise ValueError("Preprocessing produced NaN or infinite values; inspect the dataset.")

    model = build_autoencoder(x_train.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=32,
        validation_split=0.10,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=1,
    )

    calibration_errors = reconstruction_errors(model, x_calibration)
    threshold = float(np.percentile(calibration_errors, threshold_quantile))
    test_errors = reconstruction_errors(model, x_test)
    test_predictions = (test_errors > threshold).astype(np.int8)
    metrics = calculate_metrics(test_labels, test_errors, test_predictions)
    metrics.update(
        {
            "threshold": threshold,
            "threshold_quantile": float(threshold_quantile),
            "threshold_source": "normal-only calibration partition",
            "training_normal_records": int(len(normal_training_indices)),
            "calibration_normal_records": int(len(normal_calibration_indices)),
            "original_feature_count": int(len(feature_columns)),
            "model_feature_count_after_empty_column_removal": int(len(available_columns)),
            "encoded_feature_count": int(x_train.shape[1]),
            "seed": int(seed),
        }
    )
    save_artifacts(
        model=model,
        preprocessor=preprocessor,
        feature_columns=available_columns,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        calibration_count=len(normal_calibration_indices),
        normal_label=normal_token,
        metrics=metrics,
        seed=seed,
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def find_dataset() -> str | None:
    """Find the bundled CSV in one of the documented locations."""
    for candidate in DATASET_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV path; defaults to data/bin_data.csv or bin_data.csv")
    parser.add_argument(
        "--normal-label",
        help="Exact label value for normal traffic; inferred from common names when omitted",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=99.0,
        help="Percentile of normal-only calibration error used as the threshold (default: 99)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    target_path = args.data or find_dataset()
    if target_path is None:
        parser.error("Dataset not found. Extract bin_data.rar to data/bin_data.csv or pass --data.")
    try:
        train_pipeline(
            target_path,
            normal_label=args.normal_label,
            threshold_quantile=args.threshold_quantile,
            seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
