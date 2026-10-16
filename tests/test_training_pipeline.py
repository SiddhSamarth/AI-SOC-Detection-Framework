"""Integration test for fitting, calibrating, evaluating, and scoring synthetic data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_autoencoder
from score_autoencoder import score_pipeline


class TrainingPipelineIntegrationTests(unittest.TestCase):
    def test_training_uses_separate_calibration_and_test_artifacts(self):
        rng = np.random.default_rng(42)
        n = 120
        labels = np.array(["normal"] * (n // 2) + ["abnormal"] * (n // 2))
        normal_bytes = rng.normal(100, 8, n // 2)
        abnormal_bytes = rng.normal(220, 12, n // 2)
        normal_duration = rng.normal(1.0, 0.15, n // 2)
        abnormal_duration = rng.normal(5.0, 0.7, n // 2)
        data = pd.DataFrame(
            {
                "bytes": np.concatenate([normal_bytes, abnormal_bytes]),
                "duration": np.concatenate([normal_duration, abnormal_duration]),
                "protocol": ["tcp", "udp"] * (n // 4) + ["tcp", "icmp"] * (n // 4),
                "label": labels,
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "sample.csv"
            output = root / "scores.csv"
            data.to_csv(dataset, index=False)
            model_directory = root / "models"

            with patch.object(train_autoencoder, "MODEL_DIRECTORY", model_directory):
                metrics = train_autoencoder.train_pipeline(
                    dataset, normal_label="normal", threshold_quantile=99.0, seed=42
                )

            for artifact in (
                "autoencoder.keras",
                "preprocessor.pkl",
                "feature_columns.json",
                "threshold.json",
                "evaluation.json",
            ):
                self.assertTrue((model_directory / artifact).is_file(), artifact)

            threshold = json.loads((model_directory / "threshold.json").read_text(encoding="utf-8"))
            self.assertEqual(threshold["method"], "quantile of normal-only calibration reconstruction errors")
            self.assertGreaterEqual(threshold["calibration_normal_records"], 5)
            self.assertEqual(metrics["test_records"], 24)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("false_positive_rate", metrics)
            self.assertEqual(metrics["threshold_source"], "normal-only calibration partition")

            with patch.object(train_autoencoder, "MODEL_DIRECTORY", model_directory):
                self.assertEqual(score_pipeline(dataset, output), 0)
            scored = pd.read_csv(output)
            self.assertEqual(len(scored), len(data))
            self.assertIn("reconstruction_error", scored.columns)
            self.assertIn("threshold", scored.columns)
            self.assertIn("predicted_label", scored.columns)


if __name__ == "__main__":
    unittest.main()
