"""Score telemetry with the trained autoencoder and flag high-reconstruction-error records."""
import os
import json
import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
from train_autoencoder import find_dataset


def load_artifacts():
    """Load the model, scaler, label encoder and threshold saved by training."""
    # Inference only needs predict(); skipping compile avoids Keras 3 failing to
    # deserialize the legacy 'mse' loss stored in the HDF5 file.
    model = load_model("models/autoencoder.h5", compile=False)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("models/threshold.json", "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]
    return model, scaler, le, threshold


def preprocess(df, scaler):
    """Apply the fitted scaler to the raw feature frame."""
    return scaler.transform(df)


def compute_scores(model, features):
    """Return the per-record reconstruction MSE for ``features``."""
    recon = model.predict(features)
    return np.mean(np.square(features - recon), axis=1)


def classify(errors, threshold):
    """Label records whose error exceeds ``threshold`` as anomalies (1), else 0."""
    return (errors > threshold).astype(int)


def score_pipeline(data_path):
    """Score ``data_path`` and write per-record results to ``anomaly_scores.csv``."""
    if not os.path.exists("models/autoencoder.h5"):
        print(
            "Error: Trained model 'models/autoencoder.h5' not found. "
            "Please run train_autoencoder.py first."
        )
        return

    print(f"Loading trained artifacts and scoring {data_path}...")
    model, scaler, _, threshold = load_artifacts()
    df = pd.read_csv(data_path)
    df.ffill(inplace=True)
    raw = df.drop("label", axis=1) if "label" in df.columns else df
    features = preprocess(raw, scaler)
    errors = compute_scores(model, features)
    preds = classify(errors, threshold)
    df["reconstruction_error"] = errors
    df["predicted_label"] = preds
    df.to_csv("anomaly_scores.csv", index=False)
    print(f"Scoring complete. Flagged {preds.sum()} anomalies out of {len(preds)} total events.")
    print("Results saved to anomaly_scores.csv.")


if __name__ == "__main__":
    target_path = find_dataset()
    if target_path:
        score_pipeline(target_path)
    else:
        print("Error: Dataset not found.")
        print(
            "Please uncompress 'bin_data.rar' into 'data/bin_data.csv' or 'bin_data.csv' "
            "before running score_autoencoder.py."
        )
