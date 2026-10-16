"""Unit tests for label handling, partitioning, preprocessing, and scoring schema."""
import unittest

import numpy as np
import pandas as pd

from score_autoencoder import prepare_features
from train_autoencoder import make_preprocessor, map_labels, split_indices


class LabelMappingTests(unittest.TestCase):
    def test_maps_normal_to_zero_and_other_classes_to_one(self):
        labels = pd.Series(["normal", "normal", "DoS", "probe", "abnormal"])
        mapped, normal_label = map_labels(labels)
        self.assertEqual(normal_label, "normal")
        self.assertEqual(mapped.tolist(), [0, 0, 1, 1, 1])

    def test_accepts_explicit_normal_label(self):
        labels = pd.Series(["BENIGN", "attack", "attack"])
        mapped, normal_label = map_labels(labels, normal_label="BENIGN")
        self.assertEqual(normal_label, "benign")
        self.assertEqual(mapped.tolist(), [0, 1, 1])

    def test_refuses_to_guess_unknown_normal_label(self):
        with self.assertRaisesRegex(ValueError, "Could not safely infer"):
            map_labels(pd.Series(["clean-traffic", "intrusion", "intrusion"]))


class SplitTests(unittest.TestCase):
    def test_partitions_are_disjoint_and_stratified(self):
        labels = np.array([0] * 100 + [1] * 100)
        train, calibration, test = split_indices(labels, seed=7)
        self.assertEqual(len(set(train) & set(calibration)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(calibration) & set(test)), 0)
        self.assertEqual(len(train) + len(calibration) + len(test), len(labels))
        self.assertEqual(len(test), 40)
        for part in (train, calibration, test):
            self.assertEqual(set(np.unique(labels[part])), {0, 1})

    def test_split_is_reproducible(self):
        labels = np.array([0] * 50 + [1] * 50)
        first = split_indices(labels, seed=42)
        second = split_indices(labels, seed=42)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_small_or_single_class_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            split_indices(np.array([0] * 4 + [1] * 20))


class PreprocessingTests(unittest.TestCase):
    def test_numeric_and_categorical_features_and_unknown_categories(self):
        training = pd.DataFrame(
            {
                "bytes": [10.0, 20.0, np.nan, 40.0],
                "protocol": ["tcp", "udp", "tcp", "udp"],
            }
        )
        preprocessor = make_preprocessor(training)
        transformed = preprocessor.fit_transform(training)
        self.assertTrue(np.isfinite(transformed).all())
        unseen = pd.DataFrame({"bytes": [25.0], "protocol": ["unknown-protocol"]})
        scored = preprocessor.transform(unseen)
        self.assertEqual(scored.shape[1], transformed.shape[1])
        self.assertTrue(np.isfinite(scored).all())

    def test_scoring_reorders_features_and_ignores_label(self):
        frame = pd.DataFrame(
            {"label": ["normal"], "protocol": ["tcp"], "bytes": [17.0], "extra": ["ignored"]}
        )
        result = prepare_features(frame, ["bytes", "protocol"])
        self.assertEqual(list(result.columns), ["bytes", "protocol"])
        self.assertEqual(result.iloc[0]["bytes"], 17.0)

    def test_scoring_rejects_missing_trained_features(self):
        frame = pd.DataFrame({"bytes": [17.0]})
        with self.assertRaisesRegex(ValueError, "missing required feature"):
            prepare_features(frame, ["bytes", "protocol"])


if __name__ == "__main__":
    unittest.main()
