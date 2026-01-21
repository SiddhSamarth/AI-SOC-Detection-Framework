import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError

def load_data(path):
    data = pd.read_csv(path)
    data.fillna(method='ffill', inplace=True)
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
    model.save("models/autoencoder.h5")
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)

def train_pipeline(data_path):
    data = load_data(data_path)
    data, le = encode_labels(data)
    X = data.drop("label", axis=1)
    y = data["label"]
    X_scaled, scaler = scale_features(X)
    X_train, X_test, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = build_autoencoder(X_train.shape[1])
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))
    threshold = compute_threshold(model, X_test)
    save_artifacts(model, scaler, le, threshold)

if __name__ == "__main__":
    train_pipeline("data/bin_data.csv")
