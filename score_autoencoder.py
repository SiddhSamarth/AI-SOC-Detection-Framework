"""Score network telemetry with a trained normal-only autoencoder."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from train_autoencoder import LABEL_COLUMN, MODEL_DIRECTORY, find_dataset


def load_artifacts():
    """Load only the model and preprocessing artifacts created by training."""
    model_path = MODEL_DIRECTORY / "autoencoder.keras"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train_autoencoder.py first."
        )
    model = load_model(model_path, compile=False)
    with (MODEL_DIRECTORY / "preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)
    with (MODEL_DIRECTORY / "feature_columns.json").open("r", encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    with (MODEL_DIRECTORY / "threshold.json").open("r", encoding="utf-8") as handle:
        threshold_info = json.load(handle)
    threshold = float(threshold_info["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Stored threshold is not finite.")
    return model, preprocessor, feature_columns, threshold


def prepare_features(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Validate and order scoring features to match the training schema."""
    matching_label = [
        column for column in data.columns if str(column).strip().lower() == LABEL_COLUMN
    ]
    features = data.drop(columns=matching_label, errors="ignore").copy()
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Scoring input is missing required feature columns: {missing}")
    extra = [column for column in features.columns if column not in feature_columns]
    if extra:
        print(f"Ignoring {len(extra)} feature column(s) not used during training.", file=sys.stderr)
    features = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    if features.empty:
        raise ValueError("The scoring input contains no records.")
    return features


def score_pipeline(data_path: str | Path, output_path: str | Path = "anomaly_scores.csv") -> int:
    """Score records, preserving the source rows and adding error/decision columns."""
    raw_data = pd.read_csv(data_path)
    if raw_data.empty:
        raise ValueError("The scoring CSV contains no records.")
    model, preprocessor, feature_columns, threshold = load_artifacts()
    features = prepare_features(raw_data, feature_columns)
    matrix = np.asarray(preprocessor.transform(features), dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessing produced NaN or infinite values; inspect the scoring data.")

    reconstructed = model.predict(matrix, verbose=0)
    errors = np.mean(np.square(matrix - reconstructed), axis=1)
    results = raw_data.copy()
    results["reconstruction_error"] = errors
    results["threshold"] = threshold
    results["predicted_label"] = (errors > threshold).astype(np.int8)
    results.to_csv(output_path, index=False)
    print(
        f"Scored {len(results)} records; flagged {int(results['predicted_label'].sum())}. "
        f"Results saved to {output_path}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV path; defaults to data/bin_data.csv or bin_data.csv")
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"), help="Output CSV path")
    args = parser.parse_args()
    target_path = args.data or find_dataset()
    if target_path is None:
        parser.error("Dataset not found. Extract bin_data.rar to data/bin_data.csv or pass --data.")
    try:
        return score_pipeline(target_path, args.output)
    except (ValueError, OSError, KeyError, pickle.PickleError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
