# Autoencoder-Based SOC Anomaly Detection Framework

An unsupervised deep learning anomaly detection pipeline using a TensorFlow/Keras autoencoder to identify anomalous security telemetry and reduce SOC alert noise.

---

## Overview

Security Operations Centers (SOCs) process large volumes of network telemetry where rule- and signature-based detection can miss unfamiliar or non-signature deviations.

This repository implements an **unsupervised deep learning anomaly detection prototype** for tabular network connection telemetry. The neural network trains exclusively on normal baseline traffic, compressing incoming feature vectors through a bottleneck layer (64 → 32 → 64) and reconstructing them. By evaluating the Mean Squared Error (MSE) reconstruction loss against an empirical statistical threshold ($\mu + \sigma$), connection records that deviate significantly from baseline patterns are flagged for analyst review.

---

## Pipeline Summary

* **Data Preprocessing:** Cleans tabular network connection features, handles missing values via forward fill, and applies `StandardScaler` to normalize numeric features.
* **Autoencoder Training:** Trains a symmetrical dense neural network to minimize reconstruction error on normal traffic ($X_{train} \rightarrow X_{train}$).
* **Threshold Calculation:** Establishes an anomaly cutoff based on the validation loss distribution ($\text{Threshold} = \mu + \sigma$).
* **Inference & Scoring:** Scores incoming batches of connection logs, appends reconstruction error metrics, and outputs flagged records to `anomaly_scores.csv`.

---

## Features

* **Symmetrical Autoencoder Architecture:** Input($D$) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(64, ReLU) → Dense($D$, Sigmoid).
* **Unsupervised Baseline Modeling:** Requires no manual labeling of attack categories during training; models normal operational baselines.
* **Persistent Artifact Pipeline:** Serializes the trained neural network (`models/autoencoder.h5`), feature scalers (`scaler.pkl`), label encoders (`label_encoder.pkl`), and threshold cutoff (`threshold.json`).
* **CI Validation:** Includes GitHub Actions workflow (`.github/workflows/pylint.yml`) for automated Python linting and code quality validation.

---

## Technologies

* **Deep Learning Framework:** TensorFlow 2.x, Keras (`Sequential`, `Dense`, `Input`, `MeanSquaredError`)
* **Machine Learning & Preprocessing:** Scikit-Learn (`StandardScaler`, `LabelEncoder`, `train_test_split`)
* **Data Processing & Scientific Computing:** Python 3.10+, Pandas, NumPy
* **CI/CD:** GitHub Actions (Pylint)

---

## Architecture & Workflow

```
[ Raw Security Telemetry ]
           │
           ▼
[ Feature Preprocessing (StandardScaler) ]
           │
           ▼
┌──────────────────────────────────────────────┐
│       Autoencoder Bottleneck Network         │
│  Input(D) ──> Dense(64) ──> Bottleneck(32)   │
│                   │                          │
│                   ▼                          │
│  Output(D) <── Dense(64) <── Latent Vector   │
└──────────────────────────────────────────────┘
           │
           ▼
[ Reconstruction Error Calculation (MSE) ]
           │
           ├── If Loss <= Threshold (μ + σ) ──> Normal Baseline Activity
           └── If Loss >  Threshold (μ + σ) ──> Flagged Anomaly (SOC Alert)
```

---

## Repository Structure

```
AI-SOC-Detection-Framework/
├── .github/
│   └── workflows/
│       └── pylint.yml          # Automated CI linting workflow
├── bin_data.rar                # Compressed security telemetry dataset
├── requirements.txt            # Python package dependencies
├── train_autoencoder.py        # Model training and threshold calculation pipeline
├── score_autoencoder.py        # Telemetry scoring and anomaly classification script
└── README.md                   # Technical documentation and operational guide
```

---

## Setup & Usage

### 1. Prerequisites & Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/SiddhSamarth/AI-SOC-Detection-Framework.git
cd AI-SOC-Detection-Framework

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

The training and evaluation dataset is packaged in `bin_data.rar` (containing tabular network connection telemetry). You must extract it before running the pipeline:

* **Using 7-Zip (Windows):**
  Right-click `bin_data.rar` → Extract to `bin_data.csv` (or create a `data/` folder and extract to `data/bin_data.csv`).
* **Using UnRAR (Linux):**
  ```bash
  mkdir -p data
  unrar e bin_data.rar data/
  ```

### 3. Training the Autoencoder

Execute `train_autoencoder.py`. The script will load the telemetry, fit the scaler and encoder, train the autoencoder across 50 epochs, compute the statistical cutoff threshold, and save all artifacts in `models/`:

```bash
python train_autoencoder.py
```

### 4. Scoring Incoming Logs

To evaluate new telemetry against the trained model and output flagged anomalies:

```bash
python score_autoencoder.py
```

Results will be exported to `anomaly_scores.csv` containing the original features, `reconstruction_error`, and `predicted_label` (0 = normal, 1 = anomaly).

---

## Model Evaluation & Performance

<p align="center">
  <img width="560" alt="Training vs Validation Loss" src="https://github.com/user-attachments/assets/0f21b0c9-b39f-4d7b-b3c2-307666575401" />
</p>

* **Convergence Profile:** Training loss stabilizes near `0.809–0.810`, while validation loss stabilizes near `0.833–0.834` across 50 epochs without divergent overfitting.
* **Separation Capability:** When evaluated against benchmark validation holdouts with injected attack classes, the model achieves distinct separation between normal baseline reconstruction error and high-loss anomalous vectors.

---

## Project Status & Limitations

* **Current Status:** Functional Machine Learning Prototype.
* **Telemetry Domain:** Trained on tabular network connection telemetry; requires domain-specific feature engineering (e.g., failed logon rates, session durations, bytes transferred) when adapting to Sysmon or CloudTrail logs.
* **Threshold Drift:** In production environments, statistical thresholds should be recalculated periodically or segmented by endpoint cluster (e.g., developer workstation vs. production domain controller) to account for operational drift.

---

## Author & Contact

* **Author:** Siddh Samarth
* **GitHub:** [@SiddhSamarth](https://github.com/SiddhSamarth)
* **Portfolio:** [siddhsamarth.in](https://siddhsamarth.in)
* **LinkedIn:** [samarthsiddh](https://www.linkedin.com/in/siddhsamarth/)
