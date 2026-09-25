"""Train a dense autoencoder on tabular network telemetry and persist its artifacts.

The autoencoder is fitted on the full training split (normal and attack records
alike); the ``label`` column is excluded from the features and is not used to
filter the training data. The anomaly threshold is derived from the
reconstruction error on the held-out test split.
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input

DATASET_CANDIDATES = ("data/bin_data.csv", "bin_data.csv")


def load_data(path):
    """Load the telemetry CSV and forward-fill missing values."""
    data = pd.read_csv(path)
    data.ffill(inplace=True)
    return data


def encode_labels(data):
    """Encode the ``label`` column as integers and return the fitted encoder."""
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    return data, le


def scale_features(df):
    """Standardise the feature matrix and return it with the fitted scaler."""
    scaler = StandardScaler()
    features = scaler.fit_transform(df)
    return features, scaler


def build_autoencoder(dim):
    """Build and compile the 64-32-64 dense autoencoder for ``dim`` input features."""
    model = Sequential([
        Input(shape=(dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def compute_threshold(model, features):
    """Return mean + std of the per-record reconstruction MSE on ``features``."""
    recon = model.predict(features)
    errors = np.mean(np.square(features - recon), axis=1)
    return errors.mean() + errors.std()


def save_artifacts(model, scaler, label_encoder, threshold):
    """Persist the model, preprocessing objects and threshold under ``models/``."""
    os.makedirs("models", exist_ok=True)
    model.save("models/autoencoder.h5")
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("models/threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": float(threshold)}, f)
    print("Model and preprocessing artifacts saved successfully in models/ directory.")


def train_pipeline(data_path):
    """Run preprocessing, training, threshold calculation and artifact export."""
    print(f"Loading telemetry from {data_path}...")
    data = load_data(data_path)
    data, le = encode_labels(data)
    features = data.drop("label", axis=1)
    y = data["label"]
    features_scaled, scaler = scale_features(features)
    x_train, x_test, _, _ = train_test_split(
        features_scaled, y, test_size=0.2, random_state=42
    )

    print(
        f"Training autoencoder bottleneck architecture on {x_train.shape[0]} samples "
        f"({x_train.shape[1]} features)..."
    )
    model = build_autoencoder(x_train.shape[1])
    model.fit(x_train, x_train, epochs=50, batch_size=32, validation_data=(x_test, x_test))

    threshold = compute_threshold(model, x_test)
    print(f"Computed anomaly detection threshold (mean + std): {threshold:.6f}")
    save_artifacts(model, scaler, le, threshold)


def find_dataset():
    """Return the first existing dataset path, or ``None`` if none is present."""
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


if __name__ == "__main__":
    target_path = find_dataset()
    if target_path:
        train_pipeline(target_path)
    else:
        print("Error: Dataset not found.")
        print(
            "Please uncompress 'bin_data.rar' into 'data/bin_data.csv' or 'bin_data.csv' "
            "before running train_autoencoder.py."
        )
