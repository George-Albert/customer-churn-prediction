"""Synthetic checks for frozen scoring, June isolation and artifact integrity."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib

from src.audit_data import sha256, write_report
from src.evaluate_model import evaluate_frames, generate_evaluation
from src.train_models import BASELINE, binary_metrics, generate_training, train_frames
from tests.test_development_review import fixture


def evaluation_fixture():
    frames, contract = fixture()
    for position in (1, 2):
        extra = frames[position].loc[frames[position].customer_start_date.eq("20230501")].tail(2).copy()
        extra["customer_start_date"] = "20230601"
        extra["fk_dwh_customers"] = ["june-fallback-0", "june-fallback-1"]
        if position == 1:
            extra["partition"] = "evaluation"
        else:
            extra["noise"] = ["0", "1"]
        frames[position] = pd.concat([frames[position], extra], ignore_index=True)
    return frames, contract


class EvaluateModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames, cls.contract = evaluation_fixture()
        _, cls.selection, fitted = train_frames(*cls.frames, cls.contract)
        cls.pipeline = fitted[cls.selection["selected_candidate"]]
        cls.selection["threshold"] = .59

    def evaluate(self, frames=None, selection=None):
        return evaluate_frames(*(frames or self.frames), self.contract, self.pipeline,
                               selection or self.selection, .5)

    def test_full_coverage_and_fallback_use_frozen_threshold_and_prior(self):
        tables, summary = self.evaluate()
        scores = tables["evaluation_metrics.csv"].set_index(["candidate", "scope"])
        name = self.selection["selected_candidate"]
        self.assertEqual(scores.loc[(name, "all"), "customers"], 4)
        self.assertEqual(scores.loc[(name, "usable"), "customers"], 2)
        self.assertEqual(scores.loc[(name, "fallback"), "customers"], 2)
        self.assertEqual(scores.loc[(name, "fallback"), "tn"], 1)
        self.assertEqual(scores.loc[(name, "fallback"), "fn"], 1)
        self.assertEqual(scores.loc[(BASELINE, "all"), "tp"], 2)
        self.assertEqual(scores.loc[(BASELINE, "all"), "fp"], 2)
        self.assertEqual(summary["threshold"], .59)
        self.assertEqual(summary["fallback_probability"], 0)
        self.assertEqual(summary["usable_coverage"], .5)
        self.assertFalse(summary["refit_performed"])
        for table in tables.values():
            self.assertNotIn("fk_dwh_customers", table.columns)

    def test_evaluation_never_fits_or_selects_and_preserves_model_state(self):
        before = self.pipeline.named_steps["preprocess"].get_feature_names_out().copy()
        model_hash = joblib.hash(self.pipeline)
        with patch.object(self.pipeline, "fit", side_effect=AssertionError("fit forbidden")), \
             patch("src.train_models.choose_candidate", side_effect=AssertionError("selection forbidden")), \
             patch("src.train_models.choose_threshold", side_effect=AssertionError("threshold search forbidden")):
            self.evaluate()
        np.testing.assert_array_equal(before, self.pipeline.named_steps["preprocess"].get_feature_names_out())
        self.assertEqual(model_hash, joblib.hash(self.pipeline))

    def test_other_partitions_are_excluded_before_label_or_amount_parsing(self):
        before, summary = self.evaluate()
        frames = [frame.copy() for frame in self.frames]
        frames[0].loc[frames[0].customer_start_date.ne("20230601"), "invoice_record_mrr_sum_month_0"] = "invalid"
        frames[2].loc[frames[2].customer_start_date.ne("20230601"), "noise"] = "invalid"
        after, after_summary = self.evaluate(frames)
        self.assertEqual(summary, after_summary)
        for name in before:
            pd.testing.assert_frame_equal(before[name], after[name])

    def test_june_labels_change_metrics_without_changing_predictions_or_rules(self):
        before, _ = self.evaluate()
        frames = [frame.copy() for frame in self.frames]
        june = frames[2].customer_start_date.eq("20230601")
        frames[2].loc[june, "noise"] = "0"
        tables, summary = self.evaluate(frames)
        row = tables["evaluation_metrics.csv"].query("scope == 'all'").iloc[0]
        self.assertTrue(pd.isna(row.roc_auc))
        self.assertEqual(row.tp, 0)
        self.assertEqual(summary["threshold"], self.selection["threshold"])
        self.assertEqual(summary["fallback_probability"], self.selection["fallback"]["probability"])
        previous = before["evaluation_metrics.csv"].iloc[0]
        self.assertEqual(row.fp + row.tp, previous.fp + previous.tp)
        self.assertEqual(row.tn + row.fn, previous.tn + previous.fn)
        self.assertTrue(all(record["roc_auc"] is None for record in summary["metrics"]))

    def test_missing_targets_features_or_invalid_june_values_fail(self):
        for kind in ("targets", "features", "label", "amount", "flag", "empty"):
            frames = [frame.copy() for frame in self.frames]
            if kind in ("targets", "features"):
                i = 2 if kind == "targets" else 0
                frames[i] = frames[i].loc[~frames[i].fk_dwh_customers.eq("20230601-0")]
            elif kind == "label":
                frames[2].loc[frames[2].customer_start_date.eq("20230601"), "noise"] = "2"
            elif kind == "amount":
                frames[0].loc[frames[0].customer_start_date.eq("20230601"), "invoice_record_mrr_sum_month_0"] = "inf"
            elif kind == "flag":
                frames[1].loc[frames[1].partition.eq("evaluation"), "fallback_required"] = "unknown"
            else:
                frames[1] = frames[1].loc[frames[1].partition.ne("evaluation")]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.evaluate(frames)

    def test_all_fallback_population_keeps_coverage_and_empty_scope_is_null(self):
        frames = [frame.copy() for frame in self.frames]
        frames[0] = frames[0].loc[frames[0].customer_start_date.ne("20230601")]
        frames[1].loc[frames[1].partition.eq("evaluation"), "fallback_required"] = "True"
        tables, summary = self.evaluate(frames)
        self.assertEqual(summary["customers"], 4)
        self.assertEqual(summary["fallback_customers"], 4)
        self.assertEqual(summary["usable_coverage"], 0)
        for record in summary["metrics"]:
            if record["scope"] == "usable":
                self.assertEqual(record["customers"], 0)
                self.assertIsNone(record["accuracy"])
                self.assertIsNone(record["roc_auc"])
        self.assertEqual(tables["evaluation_metrics.csv"].iloc[0].tn, 2)

    def test_identity_overlap_and_frozen_contract_mismatch_fail(self):
        frames = [frame.copy() for frame in self.frames]
        for frame in frames:
            frame.loc[frame.fk_dwh_customers.eq("20230601-0"), "fk_dwh_customers"] = "20230101-0"
        with self.assertRaisesRegex(ValueError, "multiple"):
            self.evaluate(frames)
        selection = dict(self.selection, predictor_columns=list(reversed(self.selection["predictor_columns"])))
        with self.assertRaisesRegex(ValueError, "contract"):
            self.evaluate(selection=selection)

    def test_aggregate_scores_match_independent_per_customer_construction(self):
        features = self.frames[0].query("customer_start_date == '20230601'").copy()
        numeric = [c for c in self.contract["predictor_columns"] if c not in ("origin_group", "origin_subgroup")]
        for column in numeric:
            features[column] = pd.to_numeric(features[column].replace("", pd.NA)).astype(float)
        p = self.pipeline.predict_proba(features[self.contract["predictor_columns"]])[:, 1]
        expected = binary_metrics([1, 0, 0, 1], list(p) + [0, 0], .59)
        tables, _ = self.evaluate()
        actual = tables["evaluation_metrics.csv"].iloc[0]
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value)

    def test_artifacts_repeat_and_tampered_pipeline_is_rejected_before_loading(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "outputs") as directory:
            preparation, models, output = [Path(directory) / name for name in ("preparation", "models", "evaluation")]
            preparation.mkdir()
            for frame, name in zip(self.frames, ("training_features.csv", "training_metadata.csv", "training_targets.csv")):
                frame.to_csv(preparation / name, index=False)
            write_report(self.contract, preparation / "preparation_summary.json")
            write_report({"inputs": {}, "artifacts": {p.name: sha256(p) for p in preparation.iterdir()}},
                         preparation / "preparation_manifest.json")
            generate_training(preparation, models)
            frozen = {p.name: sha256(p) for p in models.iterdir()}
            generate_evaluation(preparation, models, output)
            before = {p.name: sha256(p) for p in output.iterdir()}
            with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("fit forbidden")):
                generate_evaluation(preparation, models, output)
            self.assertEqual(before, {p.name: sha256(p) for p in output.iterdir()})
            self.assertEqual(frozen, {p.name: sha256(p) for p in models.iterdir()})
            manifest = json.loads((output / "evaluation_manifest.json").read_text())
            for name, digest in manifest["artifacts"].items():
                self.assertEqual(sha256(output / name), digest)
            target_path = preparation / "training_targets.csv"
            original_targets = target_path.read_bytes()
            target_path.write_bytes(original_targets + b"tampered")
            with patch("src.evaluate_model.joblib.load", side_effect=AssertionError("must verify first")), \
                 self.assertRaisesRegex(ValueError, "hash mismatch"):
                generate_evaluation(preparation, models, output)
            target_path.write_bytes(original_targets)
            with (models / "selected_pipeline.joblib").open("ab") as stream:
                stream.write(b"tampered")
            with patch("src.evaluate_model.joblib.load", side_effect=AssertionError("must verify first")), \
                 self.assertRaisesRegex(ValueError, "hash mismatch"):
                generate_evaluation(preparation, models, output)

    def test_output_cannot_overwrite_inputs_or_frozen_models(self):
        for output in ("input", "outputs/models", "outputs/preparation/evaluation", "outputs"):
            with self.assertRaisesRegex(ValueError, "Evaluation output"):
                generate_evaluation("outputs/preparation", "outputs/models", output)


if __name__ == "__main__":
    unittest.main()
