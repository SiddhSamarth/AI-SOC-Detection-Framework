import pandas as pd
import numpy as np
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def load_artifacts():
    model = load_model("models/autoencoder.h5")
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("models/threshold.json") as f:
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
    model, scaler, le, threshold = load_artifacts()
    df = pd.read_csv(data_path)
    df.fillna(method='ffill', inplace=True)
    raw = df.drop("label", axis=1)
    X = preprocess(raw, scaler)
    errors = compute_scores(model, X)
    preds = classify(errors, threshold)
    df["reconstruction_error"] = errors
    df["predicted_label"] = preds
    df.to_csv("anomaly_scores.csv", index=False)

if __name__ == "__main__":
    score_pipeline("data/bin_data.csv")
