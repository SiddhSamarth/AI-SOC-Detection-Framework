# Autoencoder-Based Anomaly Detection for Cybersecurity & SOC Operations 

## TL;DR 

A production-orientated autoencoder pipeline that learns baseline behavior from security telemetry and flags anomalous events indicating possible intrusions, insider threats, or zero-day activity. Designed for integration with SIEM/SOAR platforms to reduce alert fatigue and surface high-fidelity incidents for SOC analysts.

---

## Why this project exists (Purpose) 

Modern Security Operations Centers (SOC) face high volumes of telemetry and high rates of false positives. Traditional signature-based detection misses zero-day and polymorphic attacks. This project provides an **unsupervised, behaviour-drivenbehaviour detection engine** using an autoencoder neural network that models normal system/network behavior and highlights deviations as potential security incidents.

---

## Problem it solves 

* **Detects unknown/zero-day attacks** by focusing on deviations from baseline behaviour rather than signatures.
* **Reduces alert noise** by prioritizing events with high reconstruction error and configurable risk scoring.
* **Operates without large labeled datasets**, addressing the common problem that security data is expensive or impractical to label at scale.
* **Scales to multiple telemetry types** (Sysmon, Windows Event Logs, Zeek/Suricata, OSQuery-derived features) via a standardized preprocessing pipeline.

---

## Benefits — why recruiters/teams should care 

* **Immediate SOC value:** Lowers mean time to detect (MTTD) by surfacing behaviour-based anomalies that escape signature detection.
* **Detects stealthy threats:** Effective against living-off-the-land techniques, fileless attacks, and unusual multi-stage activity.
* **Practical deployment:** Designed for integration with Splunk, Elastic, Wazuh, or Azure Sentinel as an anomaly scoring microservice.
* **Reduced analyst workload:** Prioritised anomaly scoring and human-readable explanations (reconstruction delta) enable faster triage.
* **Resume-grade engineering:** Demonstrates detection engineering, ML pipeline design, and SOC integration skills.

---

## How it works (High-level pipeline) 

1. **Ingest & normalize** telemetry into a tabular feature set (Elastic Common Schema recommended).

```
          ┌──────────────┐
          │ Raw Logs 🌐   │
          └──────┬───────┘
                 ▼
        ┌──────────────────┐
        │ Feature Builder 🔧│
        └──────┬───────────┘
                 ▼
```

2. **Preprocessing:** missing-value handling, label encoding (optional), and feature scaling (StandardScaler).

```
        ┌────────────────────────┐
        │ Preprocessing Engine ⚙️ │
        └─────────┬──────────────┘
                  ▼
```

3. **Train Autoencoder** on baseline (benign) telemetry.

```
     ┌─────────────────────────┐
     │ Autoencoder Model 🤖     │
     ├─────────────────────────┤
     │ Learns normal behavior  │
     │ through reconstruction  │
     └──────────┬──────────────┘
                ▼
```

4. **Score events**: compute reconstruction error (MSE).

```
     ┌─────────────────────────┐
     │ Error Scoring Engine 📊 │
     └──────────┬──────────────┘
                ▼
```

5. **Flag anomalies**: identify deviations and forward for SOC review.

```
    ┌────────────────────────┐
    │ Anomaly Alerts 🚨       │
    └────────────────────────┘
```

---

## Model & Technical Details

* **Model type:** Fully-connected Autoencoder (encoder bottleneck + symmetric decoder).
* **Typical architecture:** Input → Dense(64) → Dense(32) → Dense(64) → Output (input_dim).
* **Loss:** Mean Squared Error (MSE) between input and reconstruction.
* **Training:** unsupervised; X_train = X_train (reconstruction objective).
* **Anomaly threshold:** choose via percentile-based method on validation reconstruction errors (e.g., 95th–99th percentile) or using ROC on a labelledautoencoder holdout if available.

---

## Evaluation & Metrics 📈

### Confusion Matrix Visualization
<img width="734" height="587" alt="image" src="https://github.com/user-attachments/assets/b20c0944-b66d-4b26-a5f8-af4fda521a31" />

### Explanation

The confusion matrix above reflects the model's **prediction performance** after converting reconstruction errors into class predictions.

* **Top-left (11773)**: True Negatives — normal events correctly classified.
* **Top-right (0)**: False Positives — none, meaning no normal events were misclassified as anomalies.
* **Bottom-left (0)**: False Negatives — none, indicating the model did not miss any malicious/anomalous events.
* **Bottom-right (13422)**: True Positives — anomalies correctly flagged.

### Interpretation

This result indicates **near-perfect separation** between the reconstruction error distributions of normal and anomalous classes. The autoencoder successfully learned the baseline behavior and is able to discriminate deviations with high accuracy.

Such a profile is extremely valuable in SOC operations because it:

* Reduces false alarms
* Detects anomalous or malicious behavior reliably
* Allows analysts to trust model-driven alerting workflows

---

* **Primary signal:** reconstruction error distribution (visual inspection + statistical thresholding).
* **Metrics (when labels are available):** Precision, Recall, F1-score, AUC-ROC calculated by converting reconstruction errors into binary predictions.
* **Operational metrics:** Alert volume reduction, MTTD improvement, analyst time saved (qualitative/quantitative in production).

---

## Integration & Deployment 🛰️

**Integration options:**

* Expose a lightweight REST API (FastAPI) that accepts telemetry vectors and returns anomaly score + reconstruction details.
* Batch scoring via scheduled jobs that annotate SIEM indices (e.g., Elastic ingest pipeline).
* Wrap as a Docker container for portability; use Kubernetes for scale.

**Suggested architecture:**

* Data collection (ETL from endpoint agents) → Preprocessor (feat. engineering) → Model scoring service → SIEM enrichment & alerting → Analyst dashboard / SOAR playbooks.

---

## Quickstart (How recruiters can validate locally)

1. Prepare a CSV `bin_data.csv` where the final column is optional `label` and the other columns are numeric features derived from telemetry.
2. Run the training script (example filename: `train_autoencoder.py`) which trains and outputs a model and a `threshold.json`.
3. Run the scoring script (example filename: `score_batch.py`) to produce `anomaly_scores.csv` with columns: `record_id`, `reconstruction_error`, `is_anomaly`.

*Note: include the training and scoring scripts in the repo for immediate reproduction.*

---

## Security & Privacy 🔐

* Avoid storing sensitive raw telemetry (PII) in public repos. Sanitize datasets before uploading.
* When deployed in production, secure model endpoints with mutual TLS and API keys; follow least-privilege data access.

---

## Future work & roadmap 🗺️

* **LLM-assisted explanations:** convert reconstruction deltas to human-readable hypotheses (e.g., "suspicious PowerShell command args").
* **Graph-based enrichment:** correlate anomalies across hosts/users with a SOC knowledge graph.
* **Adaptive thresholding:** dynamic thresholds per-host or per-user baseline.
* **Hybrid models:** combine autoencoder signals with supervised classifiers for prioritised, labeled incidents.

