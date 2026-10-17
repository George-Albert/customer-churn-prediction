"""Synthetic checks for holdout isolation, priors and aggregate reproducibility."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.audit_data import sha256, write_report
from src.development_review import generate_review, review_frames
from src.prepare_data import prepare_frames
from tests.test_audit_data import normalized, record


def fixture():
    rows = []
    populations = [("20230101", [1, 1, 1, 0], [0, 0]),
                   ("20230501", [1, 0], [1, 0, 0]), ("20230601", [1, 0], [])]
    for cohort, usable_labels, fallback_labels in populations:
        for index, label in enumerate(usable_labels + fallback_labels):
            rows.append(record(FK_DWH_CUSTOMERS=f"{cohort}-{index}", CUSTOMER_START_DATE=cohort,
                               FK_DATE_INVOICE=cohort if index < len(usable_labels) else "20231201",
                               noise="True" if label else "False"))
    submission = normalized([record(FK_DWH_CUSTOMERS="submission")]).drop(columns="noise")
    tables, contract = prepare_frames(normalized(rows), submission)
    # Match the documented CSV reload: IDs/tokens string, blank numerical cells missing.
    frames = [tables[name].astype("string").fillna("") for name in
              ("training_features.csv", "training_metadata.csv", "training_targets.csv")]
    return frames, contract


class DevelopmentReviewTests(unittest.TestCase):
    def test_populations_and_priors_use_correct_denominators(self):
        frames, contract = fixture()
        tables, summary = review_frames(*frames, contract)
        classes = tables["class_distribution.csv"].set_index(["partition", "scope"])
        self.assertEqual(classes.loc[("learning", "all"), "customers"], 6)
        self.assertEqual(classes.loc[("learning", "usable"), "churn_rate"], .75)
        self.assertEqual(classes.loc[("learning", "fallback"), "churn_rate"], 0)
        comparison = tables["fallback_comparison.csv"].set_index("policy")
        self.assertEqual(comparison.loc["global_learning_prior", "probability"], .5)
        self.assertEqual(comparison.loc["learning_fallback_prior", "probability"], 0)
        self.assertEqual(comparison.loc["global_learning_prior", "accuracy"], 1 / 3)
        self.assertEqual(comparison.loc["learning_fallback_prior", "accuracy"], 2 / 3)
        self.assertEqual(summary["balanced_weights_usable_learning"], {"0": 2., "1": 2 / 3})
        self.assertFalse(summary["model_fitted"])

    def test_june_outcomes_and_features_cannot_change_review(self):
        frames, contract = fixture()
        before_tables, before_summary = review_frames(*frames, contract)
        features, metadata, targets = [frame.copy() for frame in frames]
        june = features.customer_start_date.eq("20230601")
        features.loc[june, "invoice_record_mrr_sum_month_0"] = "invalid-held-out-amount"
        targets.loc[targets.customer_start_date.eq("20230601"), "noise"] = "invalid-held-out-label"
        after_tables, after_summary = review_frames(features, metadata, targets, contract)
        self.assertEqual(before_summary, after_summary)
        for name in before_tables:
            pd.testing.assert_frame_equal(before_tables[name], after_tables[name])
        self.assertNotIn("20230601", set(after_tables["cohort_distribution.csv"].cohort))

    def test_development_labels_do_not_change_learning_priors(self):
        frames, contract = fixture()
        before, _ = review_frames(*frames, contract)
        frames[2].loc[frames[2].customer_start_date.eq("20230501"), "noise"] = "0"
        after, _ = review_frames(*frames, contract)
        self.assertEqual(list(before["fallback_comparison.csv"].probability),
                         list(after["fallback_comparison.csv"].probability))

    def test_missing_month_is_not_filled_with_zero_and_na_is_literal(self):
        frames, contract = fixture()
        tables, _ = review_frames(*frames, contract)
        stats = tables["numeric_distributions.csv"]
        row = stats.loc[stats.partition.eq("learning") & stats.feature.eq("invoice_record_mrr_sum_month_1")].iloc[0]
        self.assertEqual(row["missing"], 4)
        self.assertTrue(pd.isna(row["mean"]))
        self.assertEqual(set(tables["categorical_distributions.csv"].category), {"NA"})

    def test_rejects_feature_coverage_mismatch(self):
        frames, contract = fixture()
        frames[0] = frames[0].iloc[1:]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            review_frames(*frames, contract)

    def test_negative_and_tiny_features_are_not_clipped(self):
        frames, contract = fixture()
        frames[0].loc[0, "invoice_record_mrr_sum_month_0"] = "-10"
        frames[0].loc[1, "invoice_record_mrr_sum_month_0"] = "0.0009"
        tables, _ = review_frames(*frames, contract)
        stats = tables["numeric_distributions.csv"]
        row = stats.loc[stats.partition.eq("learning") & stats.feature.eq("invoice_record_mrr_sum_month_0")].iloc[0]
        self.assertEqual(row["min"], -10)
        self.assertAlmostEqual(row["median"], (10.5 + .0009) / 2)
        for table in tables.values():
            self.assertNotIn("fk_dwh_customers", table.columns)

    def test_rejects_duplicate_linkage_keys(self):
        frames, contract = fixture()
        frames[2] = pd.concat([frames[2], frames[2].iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            review_frames(*frames, contract)

    def test_customer_cannot_cross_learning_and_development_cohorts(self):
        frames, contract = fixture()
        for frame in frames:
            frame.loc[frame.fk_dwh_customers.eq("20230501-0"), "fk_dwh_customers"] = "20230101-0"
        with self.assertRaisesRegex(ValueError, "multiple review cohorts"):
            review_frames(*frames, contract)

    def test_rejects_wrong_role_and_selected_invalid_label(self):
        for column, value, expected in (("partition", "development", "Cohort/partition"),
                                        ("noise", "2", "binary")):
            frames, contract = fixture()
            frames[1 if column == "partition" else 2].loc[0, column] = value
            with self.assertRaisesRegex(ValueError, expected):
                review_frames(*frames, contract)

    def test_invalid_and_infinite_learning_amounts_fail(self):
        for value in ("invalid", "inf"):
            frames, contract = fixture()
            frames[0].loc[0, "invoice_record_mrr_sum_month_0"] = value
            with self.assertRaises(ValueError):
                review_frames(*frames, contract)

    def test_missing_conditional_population_is_explicit(self):
        frames, contract = fixture()
        metadata = frames[1]
        frames[1] = metadata.loc[~(metadata.partition.eq("learning") & metadata.fallback_required.eq("True"))]
        tables, _ = review_frames(*frames, contract)
        self.assertEqual(list(tables["fallback_comparison.csv"].policy), ["global_learning_prior"])

    def test_repeatability_and_tampered_preparation_rejection(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "outputs") as directory:
            preparation = Path(directory) / "preparation"
            output = Path(directory) / "review"
            preparation.mkdir()
            frames, contract = fixture()
            for frame, name in zip(frames, ("training_features.csv", "training_metadata.csv", "training_targets.csv")):
                frame.to_csv(preparation / name, index=False)
            write_report(contract, preparation / "preparation_summary.json")
            write_report({"inputs": {}, "artifacts": {path.name: sha256(path) for path in preparation.iterdir()}},
                         preparation / "preparation_manifest.json")
            generate_review(preparation, output)
            before = {path.name: sha256(path) for path in output.iterdir()}
            generate_review(preparation, output)
            self.assertEqual(before, {path.name: sha256(path) for path in output.iterdir()})
            manifest = json.loads((output / "review_manifest.json").read_text())
            self.assertEqual(set(manifest["artifacts"]), set(before) - {"review_manifest.json"})
            with (preparation / "training_targets.csv").open("a") as stream:
                stream.write("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                generate_review(preparation, output)

    def test_outputs_must_stay_local_and_separate(self):
        with self.assertRaisesRegex(ValueError, "Review output"):
            generate_review("outputs/preparation", "input")
        with self.assertRaisesRegex(ValueError, "Review output"):
            generate_review("outputs/preparation", "outputs/preparation/review")


if __name__ == "__main__":
    unittest.main()
