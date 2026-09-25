import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError

def load_data(path):
    data = pd.read_csv(path)
    data.ffill(inplace=True)
    return data

def encode_labels(data):
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    return data, le

def scale_features(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    return X, scaler

def build_autoencoder(dim):
    model = Sequential([
        Input(shape=(dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def compute_threshold(model, X):
    recon = model.predict(X)
    errors = np.mean(np.square(X - recon), axis=1)
    return errors.mean() + errors.std()

def save_artifacts(model, scaler, label_encoder, threshold):
    os.makedirs("models", exist_ok=True)
    model.save("models/autoencoder.h5")
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    print("Model and preprocessing artifacts saved successfully in models/ directory.")

def train_pipeline(data_path):
    print(f"Loading telemetry from {data_path}...")
    data = load_data(data_path)
    data, le = encode_labels(data)
    X = data.drop("label", axis=1)
    y = data["label"]
    X_scaled, scaler = scale_features(X)
    X_train, X_test, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training autoencoder bottleneck architecture on {X_train.shape[0]} samples ({X_train.shape[1]} features)...")
    model = build_autoencoder(X_train.shape[1])
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))
    
    threshold = compute_threshold(model, X_test)
    print(f"Computed anomaly detection threshold (mean + std): {threshold:.6f}")
    save_artifacts(model, scaler, le, threshold)

if __name__ == "__main__":
    target_path = "data/bin_data.csv" if os.path.exists("data/bin_data.csv") else ("bin_data.csv" if os.path.exists("bin_data.csv") else None)
    if target_path:
        train_pipeline(target_path)
    else:
        print("Error: Dataset not found.")
        print("Please uncompress 'bin_data.rar' into 'data/bin_data.csv' or 'bin_data.csv' before running train_autoencoder.py.")
