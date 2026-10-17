"""Synthetic regression coverage for full-label refit and complete binary export."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.audit_data import sha256, write_report
from src.finalize_submission import aggregate_submission, finalize_frames, generate_submission, validate_submission
from src.prepare_data import KEYS, prepare_frames
from src.train_models import candidates, preprocessing_state
from tests.test_audit_data import normalized, record


def fixture():
    rows = []
    for cohort, category, amount in (("20230101", "base", "1"), ("20230501", "middle", "10"),
                                     ("20230601", "late", "100")):
        for i, label in enumerate(("True", "False")):
            rows.append(record(FK_DWH_CUSTOMERS=f"{cohort}-{i}", CUSTOMER_START_DATE=cohort,
                               FK_DATE_INVOICE=cohort, PRODUCT_SUBCATEGORY=category, mrr=amount, noise=label))
    rows += [record(FK_DWH_CUSTOMERS="fallback-jan", FK_DATE_INVOICE="20231201", noise="False"),
             record(FK_DWH_CUSTOMERS="fallback-june", CUSTOMER_START_DATE="20230601",
                    FK_DATE_INVOICE="20231201", noise="True")]
    validation = normalized([
        record(FK_DWH_CUSTOMERS="one", CUSTOMER_START_DATE="20240101", FK_DATE_INVOICE="20240115",
               PRODUCT_SUBCATEGORY="submission-only", ORIGIN_GROUP="unseen-channel"),
        record(FK_DWH_CUSTOMERS="two", CUSTOMER_START_DATE="20240101", FK_DATE_INVOICE="20240115"),
        record(FK_DWH_CUSTOMERS="two", CUSTOMER_START_DATE="20240201", FK_DATE_INVOICE="20240215", ORIGIN_GROUP="Organic"),
        record(FK_DWH_CUSTOMERS="none", CUSTOMER_START_DATE="20240101", FK_DATE_INVOICE="20240915"),
    ]).drop(columns="noise")
    train = normalized(rows)
    _, contract = prepare_frames(train, validation)
    numeric = [c for c in contract["predictor_columns"] if c not in ("origin_group", "origin_subgroup")]
    selected = candidates(numeric)["forest_unweighted"]
    selection = {"selected_candidate": "forest_unweighted", "threshold": .59,
                 "selected_parameters": selected.named_steps["classifier"].get_params(deep=False),
                 "predictor_columns": contract["predictor_columns"],
                 "fallback": {"policy": "learning_fallback_prior", "probability": 0.}}
    return train, validation, selection


class FinalizeSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.validation, cls.selection = fixture()
        cls.tables, cls.summary, cls.model = finalize_frames(cls.train, cls.validation, cls.selection)

    def test_refit_uses_all_labeled_rows_and_reestimates_conditional_prior(self):
        self.assertEqual(self.summary["fitting_customers"], 6)
        self.assertEqual(self.summary["eligible_labeled_customers"], 8)
        self.assertEqual(self.summary["fallback"]["probability"], .5)
        self.assertEqual(self.summary["fallback"]["labeled_customers"], 2)
        state = preprocessing_state(self.model)
        position = state["numeric_columns"].index("invoice_record_mrr_sum_month_0")
        self.assertEqual(state["numeric_imputation"][position], 10)
        self.assertEqual(self.summary["threshold"], .59)
        self.assertFalse(self.summary["selection_performed"])
        self.assertFalse(self.summary["new_holdout_evaluation"])

    def test_vocabulary_relearns_from_all_labels_and_excludes_submission(self):
        self.assertEqual(set(self.summary["product_vocabulary"].values()), {"base", "middle", "late"})
        state = preprocessing_state(self.model)
        self.assertNotIn("submission-only", str(state))
        self.assertNotIn("unseen-channel", str(state))
        _, default = prepare_frames(self.train, self.validation)
        self.assertEqual(set(default["product_vocabulary"].values()), {"base"})
        self.assertEqual(default["vocabulary_partitions"], ["learning"])

    def test_submission_input_values_do_not_change_fitted_model_or_prior(self):
        validation = self.validation.copy()
        validation["mrr"] = "999999"
        validation["origin_group"] = "new-submission-value"
        _, summary, model = finalize_frames(self.train, validation, self.selection)
        self.assertEqual(summary["fallback"], self.summary["fallback"])
        self.assertEqual(preprocessing_state(model), preprocessing_state(self.model))
        for a, b in zip(model.named_steps["classifier"].estimators_, self.model.named_steps["classifier"].estimators_):
            np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
            np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
            np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)

    def test_export_has_exact_headers_unique_ids_and_fallback_audit(self):
        p = self.tables["predictions.csv"]
        validate_submission(p, self.validation.fk_dwh_customers.unique())
        self.assertEqual(list(p.columns), ["FK_DWH_CUSTOMERS", "Predictions"])
        self.assertEqual(len(p), 3)
        self.assertEqual(len(self.tables["scoring_audit.csv"]), 4)
        fallback = self.tables["scoring_audit.csv"].query("fk_dwh_customers == 'none'").iloc[0]
        self.assertEqual(fallback.probability, .5)
        self.assertEqual(fallback.score_source, "conditional_fallback")
        self.assertEqual(self.summary["submission"]["multiple_cohort_customers"], 1)

    def test_equal_cohort_mean_precedes_threshold_including_mixed_fallback(self):
        audit = pd.DataFrame({"fk_dwh_customers": ["a", "a", "b"],
                              "customer_start_date": ["20240101", "20240201", "20240101"],
                              "probability": [.9, .1, .5], "score_source": ["model", "conditional_fallback", "model"]})
        p = aggregate_submission(audit, ["a", "b"], .59).set_index("FK_DWH_CUSTOMERS")
        self.assertEqual(p.loc["a", "Predictions"], 0)  # voting or threshold-before-average would be ambiguous
        self.assertEqual(aggregate_submission(audit, ["a", "b"], .5).Predictions.tolist(), [1, 1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_submission(pd.concat([audit, audit.iloc[:1]]), ["a", "b"], .5)

    def test_invalid_export_contract_and_probability_fail(self):
        p = self.tables["predictions.csv"]
        broken = [p.rename(columns={"Predictions": "probability"}), pd.concat([p, p.iloc[:1]]),
                  p.assign(Predictions=2), p.iloc[:1], p.assign(FK_DWH_CUSTOMERS="")]
        for frame in broken:
            with self.assertRaises(ValueError):
                validate_submission(frame, self.validation.fk_dwh_customers.unique())
        audit = self.tables["scoring_audit.csv"].copy()
        audit.loc[0, "probability"] = np.nan
        with self.assertRaises(ValueError):
            aggregate_submission(audit, self.validation.fk_dwh_customers.unique(), .59)

    def test_frozen_parameters_and_policy_cannot_be_silently_changed(self):
        for change in ({"selected_candidate": "unknown"}, {"fallback": {"policy": "global"}},
                       {"selected_parameters": dict(self.selection["selected_parameters"], max_depth=100)},
                       {"threshold": -1}):
            with self.assertRaises(ValueError):
                finalize_frames(self.train, self.validation, dict(self.selection, **change))
        with patch("src.train_models.choose_candidate", side_effect=AssertionError("no new selection")), \
             patch("src.train_models.choose_threshold", side_effect=AssertionError("no new threshold")):
            finalize_frames(self.train, self.validation, self.selection)

    def test_invalid_vocabulary_roles_and_empty_fallback_fail(self):
        for roles in ((), ("submission",)):
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                prepare_frames(self.train, self.validation, vocabulary_partitions=roles)
        train = self.train.loc[~self.train.fk_dwh_customers.str.startswith("fallback")]
        with self.assertRaisesRegex(ValueError, "Conditional fallback"):
            finalize_frames(train, self.validation, self.selection)

    def test_artifacts_repeat_round_trip_and_frozen_sources_are_protected(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "outputs") as directory:
            input_dir, models, evaluation, output = [Path(directory) / n for n in ("input", "models", "evaluation", "submission")]
            for d in (input_dir, models, evaluation):d.mkdir()
            # Reverse the explicit normalization map to build synthetic source-format CSVs.
            from src.audit_data import COLUMN_MAP
            inverse = {value: key for key, value in COLUMN_MAP.items()}
            self.train.rename(columns=inverse).to_csv(input_dir / "training.csv", index=False)
            self.validation.rename(columns=inverse).to_csv(input_dir / "validation.csv", index=False)
            (input_dir / "case_study.ipynb").write_text('{}')
            originals = {p.name: sha256(p) for p in input_dir.iterdir()}
            write_report(self.selection, models / "selection.json")
            joblib.dump(self.model, models / "selected_pipeline.joblib")
            import importlib.metadata, platform
            write_report({"original_inputs": originals, "artifacts": {p.name: sha256(p) for p in models.iterdir()},
                          "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "joblib")},
                          "python": platform.python_version()}, models / "model_manifest.json")
            write_report({"selected_candidate": self.selection["selected_candidate"], "threshold": .59,
                          "fallback_probability": 0., "refit_performed": False, "selection_performed": False},
                         evaluation / "evaluation_summary.json")
            write_report({"model_manifest_sha256": sha256(models / "model_manifest.json"),
                          "original_inputs": originals,
                          "artifacts": {p.name: sha256(p) for p in evaluation.iterdir()}}, evaluation / "evaluation_manifest.json")
            frozen = {str(p): sha256(p) for d in (models, evaluation, input_dir) for p in d.iterdir()}
            generate_submission(input_dir, models, output, evaluation)
            before = {p.name: sha256(p) for p in output.iterdir()}
            generate_submission(input_dir, models, output, evaluation)
            self.assertEqual(before, {p.name: sha256(p) for p in output.iterdir()})
            self.assertEqual(frozen, {str(p): sha256(p) for d in (models, evaluation, input_dir) for p in d.iterdir()})
            restored = joblib.load(output / "final_pipeline.joblib")
            self.assertEqual(preprocessing_state(restored), preprocessing_state(self.model))
            manifest = json.loads((output / "submission_manifest.json").read_text())
            for name, digest in manifest["artifacts"].items():self.assertEqual(sha256(output / name), digest)
            with (input_dir / "training.csv").open("a") as stream:stream.write('tampered')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                generate_submission(input_dir, models, output, evaluation)

    def test_output_cannot_overwrite_other_stages(self):
        for output in ("input", "outputs/models/submission", "outputs/evaluation", "outputs/preparation", "outputs"):
            with self.assertRaisesRegex(ValueError, "Submission output"):
                generate_submission("input", "outputs/models", output)


if __name__ == "__main__":
    unittest.main()
