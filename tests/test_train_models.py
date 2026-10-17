"""Synthetic regression tests for fitting isolation, selection and full coverage."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.audit_data import sha256, write_report
from src.train_models import (BASELINE, binary_metrics, choose_candidate, choose_threshold,
                              generate_training, preprocessing_state, train_frames)
from tests.test_development_review import fixture


def model_fixture():
    frames, contract = fixture()
    features = frames[0]
    learning = features.customer_start_date.eq("20230101")
    features.loc[learning, "invoice_record_mrr_sum_month_0"] = ["", "1", "3", "100"]
    development = features.customer_start_date.eq("20230501")
    features.loc[development, "invoice_record_mrr_sum_month_0"] = "1000000"
    features.loc[development, "origin_subgroup"] = "new-development-category"
    return frames, contract


class TrainModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames, cls.contract = model_fixture()
        cls.tables, cls.selection, cls.fitted = train_frames(*cls.frames, cls.contract)

    def test_six_variants_and_constant_majority_reference_have_full_coverage(self):
        comparison = self.tables["comparison_metrics.csv"]
        self.assertEqual(len(self.fitted), 7)
        self.assertEqual(set(comparison.scope), {"all", "usable", "fallback"})
        self.assertTrue(comparison.loc[comparison.scope.eq("all"), "customers"].eq(5).all())
        self.assertTrue(comparison.loc[comparison.scope.eq("usable"), "customers"].eq(2).all())
        self.assertTrue(comparison.loc[comparison.scope.eq("fallback"), "customers"].eq(3).all())
        baseline = comparison.loc[comparison.candidate.eq(BASELINE) & comparison.scope.eq("all")].iloc[0]
        self.assertEqual(baseline.roc_auc, .5)
        self.assertEqual(self.fitted[BASELINE].class_prior_.tolist(), [.5, .5])
        self.assertEqual(self.selection["balanced_weights"], {"0": 2., "1": 2 / 3})

    def test_imputation_scaling_and_encoding_fit_only_learning(self):
        model = self.fitted["logistic_unweighted"]
        state = preprocessing_state(model)
        column = state["numeric_columns"].index("invoice_record_mrr_sum_month_0")
        self.assertEqual(state["numeric_imputation"][column], 3)
        self.assertEqual(state["numeric_scaling_mean"][column], 26.75)
        self.assertIn(["NA"], state["categorical_values"])
        self.assertNotIn("new-development-category", str(state["categorical_values"]))
        missing = state["numeric_columns"].index("invoice_record_mrr_sum_month_1")
        self.assertEqual(state["numeric_imputation"][missing], 0)
        self.assertIn("numeric__invoice_record_mrr_sum_month_1", state["feature_names"])
        self.assertNotIn("numeric_scaling_mean", preprocessing_state(self.fitted["tree_unweighted"]))

    def test_fallback_probability_is_conditional_and_used_for_all_real_candidates(self):
        self.assertEqual(self.selection["fallback"]["probability"], 0)
        self.assertEqual(self.selection["fallback"]["learning_customers"], 2)
        comparison = self.tables["comparison_metrics.csv"]
        subset = comparison.loc[comparison.scope.eq("fallback") & comparison.candidate.ne(BASELINE)]
        self.assertTrue(subset.tn.eq(2).all())
        self.assertTrue(subset.fn.eq(1).all())

    def test_june_invalid_labels_and_features_have_no_effect(self):
        frames = [frame.copy() for frame in self.frames]
        features, _, targets = frames
        features.loc[features.customer_start_date.eq("20230601"), "invoice_record_mrr_sum_month_0"] = "invalid"
        targets.loc[targets.customer_start_date.eq("20230601"), "noise"] = "invalid"
        tables, selection, _ = train_frames(*frames, self.contract)
        self.assertEqual(selection, self.selection)
        for name in tables:
            pd.testing.assert_frame_equal(tables[name], self.tables[name])

    def test_development_labels_never_change_fitted_models_or_prior(self):
        frames = [frame.copy() for frame in self.frames]
        targets = frames[2]
        mask = targets.customer_start_date.eq("20230501")
        targets.loc[mask, "noise"] = targets.loc[mask, "noise"].map({"0": "1", "1": "0"})
        _, selection, fitted = train_frames(*frames, self.contract)
        self.assertEqual(selection["fallback"], self.selection["fallback"])
        for name in self.fitted:
            self.assertEqual(joblib.hash(fitted[name]), joblib.hash(self.fitted[name]))

    def test_empty_learning_fallback_fails_without_silent_global_substitution(self):
        frames = [frame.copy() for frame in self.frames]
        metadata = frames[1]
        frames[1] = metadata.loc[~(metadata.partition.eq("learning") & metadata.fallback_required.eq("True"))]
        with self.assertRaisesRegex(ValueError, "Conditional fallback requires"):
            train_frames(*frames, self.contract)

    def test_model_ranking_uses_auc_then_declared_ties_and_excludes_reference(self):
        comparison = pd.DataFrame([
            {"candidate": name, "scope": "all", "roc_auc": auc, "balanced_accuracy": balanced}
            for name, auc, balanced in [(BASELINE, 1., 1.), ("forest_balanced", .8, .7),
                                        ("tree_unweighted", .8, .7), ("logistic_balanced", .8, .7),
                                        ("logistic_unweighted", .8, .7)]])
        self.assertEqual(choose_candidate(comparison), "logistic_unweighted")
        comparison.loc[comparison.candidate.eq("tree_unweighted"), "balanced_accuracy"] = .9
        self.assertEqual(choose_candidate(comparison), "tree_unweighted")
        comparison.loc[comparison.candidate.eq("forest_balanced"), "roc_auc"] = .9
        self.assertEqual(choose_candidate(comparison), "forest_balanced")

    def test_threshold_ties_prefer_nearest_half_then_lower(self):
        search = pd.DataFrame({"threshold": [.2, .4, .6, .5], "balanced_accuracy": [.8, .8, .8, .7]})
        self.assertEqual(choose_threshold(search), .4)
        search.loc[search.threshold.eq(.5), "balanced_accuracy"] = .8
        self.assertEqual(choose_threshold(search), .5)
        self.assertEqual(len(self.tables["threshold_search.csv"]), 101)

    def test_metrics_match_known_confusion_counts_and_probability_error(self):
        result = binary_metrics([0, 0, 1, 1], [.1, .8, .6, .2], .5)
        self.assertEqual([result[name] for name in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["f1"], .5)
        self.assertAlmostEqual(result["brier_score"], (.01 + .64 + .16 + .64) / 4)
        self.assertEqual(binary_metrics([0, 1], [.5, .5], .5)["tp"], 1)
        self.assertIsNone(binary_metrics([0, 0], [.1, .2], .5)["roc_auc"])

    def test_invalid_probabilities_labels_and_thresholds_fail(self):
        for y, p, t in [([0, 1], [.1], .5), ([0, 1.2], [.1, .2], .5),
                         ([0, 1], [.1, np.nan], .5), ([0, 1], [-.1, 1.1], .5),
                         ([0, 1], [.1, .2], -1)]:
            with self.assertRaises(ValueError):
                binary_metrics(y, p, t)

    def test_local_artifacts_repeat_and_selected_pipeline_round_trips(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "outputs") as directory:
            preparation, output = Path(directory) / "preparation", Path(directory) / "models"
            preparation.mkdir()
            for frame, name in zip(self.frames, ("training_features.csv", "training_metadata.csv", "training_targets.csv")):
                frame.to_csv(preparation / name, index=False)
            write_report(self.contract, preparation / "preparation_summary.json")
            write_report({"inputs": {}, "artifacts": {path.name: sha256(path) for path in preparation.iterdir()}},
                         preparation / "preparation_manifest.json")
            generate_training(preparation, output)
            before = {path.name: sha256(path) for path in output.iterdir()}
            generate_training(preparation, output)
            self.assertEqual(before, {path.name: sha256(path) for path in output.iterdir()})
            manifest = json.loads((output / "model_manifest.json").read_text())
            for name, digest in manifest["artifacts"].items():
                self.assertEqual(sha256(output / name), digest)
            restored = joblib.load(output / "selected_pipeline.joblib")
            self.assertEqual(joblib.hash(restored), joblib.hash(self.fitted[self.selection["selected_candidate"]]))
            with (preparation / "training_targets.csv").open("a") as stream:
                stream.write("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                generate_training(preparation, output)

    def test_output_path_cannot_overwrite_sources_or_preparation(self):
        for output in ("input", "outputs/preparation/models"):
            with self.assertRaisesRegex(ValueError, "Model output"):
                generate_training("outputs/preparation", output)


if __name__ == "__main__":
    unittest.main()
