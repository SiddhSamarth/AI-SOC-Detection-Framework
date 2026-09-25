import os
import json
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def load_artifacts():
    model = load_model("models/autoencoder.h5")
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("models/threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]
    return model, scaler, le, threshold

def preprocess(df, scaler):
    return scaler.transform(df)

def compute_scores(model, X):
    recon = model.predict(X)
    return np.mean(np.square(X - recon), axis=1)

def classify(errors, threshold):
    return (errors > threshold).astype(int)

def score_pipeline(data_path):
    if not os.path.exists("models/autoencoder.h5"):
        print("Error: Trained model 'models/autoencoder.h5' not found. Please run train_autoencoder.py first.")
        return
        
    print(f"Loading trained artifacts and scoring {data_path}...")
    model, scaler, le, threshold = load_artifacts()
    df = pd.read_csv(data_path)
    df.ffill(inplace=True)
    raw = df.drop("label", axis=1) if "label" in df.columns else df
    X = preprocess(raw, scaler)
    errors = compute_scores(model, X)
    preds = classify(errors, threshold)
    df["reconstruction_error"] = errors
    df["predicted_label"] = preds
    df.to_csv("anomaly_scores.csv", index=False)
    print(f"Scoring complete. Flagged {preds.sum()} anomalies out of {len(preds)} total events.")
    print("Results saved to anomaly_scores.csv.")

if __name__ == "__main__":
    target_path = "data/bin_data.csv" if os.path.exists("data/bin_data.csv") else ("bin_data.csv" if os.path.exists("bin_data.csv") else None)
    if target_path:
        score_pipeline(target_path)
    else:
        print("Error: Dataset not found.")
        print("Please uncompress 'bin_data.rar' into 'data/bin_data.csv' or 'bin_data.csv' before running score_autoencoder.py.")
