# Autoencoder-Based SOC Anomaly Detection

A small unsupervised-learning prototype for flagging unusual records in tabular network telemetry. It trains a dense autoencoder on records labelled normal, sets a reconstruction-error threshold using a separate normal-only calibration split, and reports performance on a held-out test split.

This is a research and training project. An anomaly score is a triage signal, not proof that an event is malicious.

## How it works

1. Load the CSV and identify the `label` column.
2. Create disjoint development, calibration and test partitions (60% / 20% / 20%) using stratification.
3. Keep only normal records in the model-training subset.
4. Fit missing-value handling, scaling and categorical encoding on normal training records only.
5. Train a dense autoencoder to reconstruct the preprocessed normal records. A linear output layer is used because standardised values are not limited to the interval [0, 1].
6. Set the threshold to the selected percentile of reconstruction errors from the separate normal-only calibration records. The default is the 99th percentile.
7. Evaluate once on the held-out test set, which contains normal and non-normal records, and save the metrics.
8. Reuse the saved model and preprocessor to score new CSV files with the same feature schema.

The labels are used to identify normal records for training, select normal calibration records and evaluate the final test split. They are not included as model input features.

## Repository contents

```text
AI-SOC-Detection-Framework/
├── .github/workflows/pylint.yml
├── bin_data.rar
├── train_autoencoder.py
├── score_autoencoder.py
├── requirements.txt
├── tests/
│   ├── test_pipeline.py
│   └── test_training_pipeline.py
└── README.md
```

Training writes generated artifacts to `models/`:

- `autoencoder.keras` — trained network.
- `preprocessor.pkl` — fitted imputation, scaling and encoding steps.
- `feature_columns.json` — feature names and ordering used for training.
- `threshold.json` — threshold, calibration quantile, seed and calibration sample count.
- `evaluation.json` — metrics from the held-out test partition.

Scoring writes `anomaly_scores.csv` by default.

## Setup

Python 3.10 or 3.11 is supported by the CI workflow.

```bash
git clone https://github.com/SiddhSamarth/AI-SOC-Detection-Framework.git
cd AI-SOC-Detection-Framework

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Extract `bin_data.rar` so that the CSV is located at `data/bin_data.csv` or `bin_data.csv`. The CSV must contain a `label` column and feature columns. The program attempts to infer the normal class from common labels such as `normal`, `benign`, `legitimate`, `background` or `0`. When the dataset uses a different label, pass it explicitly.

## Train and evaluate

```bash
python train_autoencoder.py --data data/bin_data.csv --normal-label normal
```

Useful options:

- `--normal-label`: exact label value for normal traffic.
- `--threshold-quantile`: percentile of normal-only calibration errors; default `99`.
- `--seed`: random seed; default `42`.

The pipeline stops with an error when the label is ambiguous, either class is too small for the split, too few normal training/calibration examples are available, or preprocessing results are not finite. Inspect the error and dataset before adjusting the constraints.

## Score records

```bash
python score_autoencoder.py --data data/bin_data.csv --output anomaly_scores.csv
```

The output preserves the input columns and adds:

- `reconstruction_error` — per-row mean squared reconstruction error.
- `threshold` — the saved cutoff used for this run.
- `predicted_label` — `0` for not flagged and `1` for flagged.

The input must contain all feature columns used during training. Extra input columns are ignored by the model and reported to stderr. If the input includes a `label` column, it is kept in the output but excluded from model features.

**Only load model and pickle artifacts from a source you trust.** Pickle files can execute code when loaded.

## Evaluation

The pipeline saves test-set metrics in `models/evaluation.json`, including:

- Precision, recall and F1 at the calibrated threshold.
- True positives, false positives, true negatives and false negatives.
- False-positive rate.
- ROC-AUC and average precision using reconstruction errors as scores.

The test partition is not used to fit preprocessing, train the model or choose the threshold. Report the final test metrics for the dataset and seed used; results may vary with a different split.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover label handling, split isolation, preprocessing of numerical and categorical columns, feature-schema checks, and an end-to-end training/scoring run on synthetic records. They do not require the bundled dataset to run, but the end-to-end test requires the listed dependencies.

## Limitations

- Reconstruction error is not a direct measure of maliciousness. Unusual legitimate traffic can be flagged, and attacks resembling normal training records can be missed.
- The 99th-percentile threshold is calibrated on a limited normal-only subset. It does not guarantee a 1% production false-positive rate, particularly when telemetry changes over time or differs between environments.
- A single threshold may be unsuitable for hosts with very different activity patterns.
- The evaluation is only as representative as the bundled dataset and split. Dataset source, licensing, class definitions and acquisition details should be recorded alongside any published result.
- Performance on novel or zero-day attacks has not been demonstrated by this project.
- Do not deploy as an autonomous blocking control without additional validation, monitoring, analyst review and rollback procedures.

## Author

Siddh Samarth  
GitHub: [@SiddhSamarth](https://github.com/SiddhSamarth)  
Portfolio: [siddhsamarth.in](https://siddhsamarth.in)  
LinkedIn: [samarthsiddh](https://www.linkedin.com/in/siddhsamarth/)
