"""Guard exploratory aggregation against silent missing-month imputation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.eda import customer_activity, distribution_table, generate_eda
from tests.test_audit_data import normalized, record


class EdaTests(unittest.TestCase):
    def test_absent_month_is_missing_not_zero(self):
        result = customer_activity(normalized([record(mrr="-2.5")]))
        self.assertEqual(result.invoice_record_mrr_sum_month_0.iloc[0], -2.5)
        self.assertTrue(pd.isna(result.invoice_record_mrr_sum_month_1.iloc[0]))

    def test_count_distinct_invoices_and_subscriptions_without_collapsing_records(self):
        rows = [record(), record(FK_SUBSCRIPTIONS="subscription-2", PRODUCT_GROUP="Domain", mrr="2")]
        result = customer_activity(normalized(rows))
        self.assertEqual(result.invoice_records.iloc[0], 2)
        self.assertEqual(result.distinct_invoices.iloc[0], 1)
        self.assertEqual(result.distinct_subscriptions.iloc[0], 2)
        self.assertEqual(result.invoice_record_mrr_sum_month_0.iloc[0], 12.5)

    def test_exploratory_subset_does_not_include_negative_tenure_or_change_source(self):
        frame = normalized([record(), record(TENURE="-1", mrr="1000")])
        original = frame.copy(deep=True)
        result = customer_activity(frame)
        self.assertEqual(result.invoice_record_mrr_sum_month_0.iloc[0], 10.5)
        pd.testing.assert_frame_equal(frame, original)

    def test_invalid_monthly_revenue_stays_missing(self):
        result = customer_activity(normalized([record(mrr="invalid")]))
        self.assertTrue(pd.isna(result.invoice_record_mrr_sum_month_0.iloc[0]))

    def test_distribution_export_does_not_expose_customer_ids(self):
        activity = customer_activity(normalized([record()]))
        result = distribution_table(activity, dataset="synthetic", scope="test")
        self.assertNotIn("customer-1", result.to_csv(index=False))

    def test_numeric_month_representations_have_same_exploratory_aggregation(self):
        result = customer_activity(normalized([record(TENURE="0.0")]))
        self.assertEqual(result.invoice_record_mrr_sum_month_0.iloc[0], 10.5)

    def test_phase_one_manifest_excludes_later_phase_files(self):
        def save_stub(fig, path):
            import matplotlib.pyplot as plt
            Path(path).write_bytes(b"synthetic figure")
            plt.close(fig)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, outputs = root / "input", root / "outputs"
            inputs.mkdir()
            (outputs / "preparation").mkdir(parents=True)
            (outputs / "preparation/training_features.csv").write_text("unrelated-phase\n")
            rows = [record(), record(TENURE="1"), record(FK_DWH_CUSTOMERS="customer-2", noise="False")]
            pd.DataFrame(rows).to_csv(inputs / "training.csv", index=False)
            pd.DataFrame(rows).drop(columns="noise").to_csv(inputs / "validation.csv", index=False)
            with patch("src.eda.save_figure", side_effect=save_stub):
                generate_eda(inputs, outputs)
            manifest = json.loads((outputs / "run_manifest.json").read_text())
            self.assertEqual(len(manifest["artifacts"]), 15)
            self.assertFalse(any(name.startswith("preparation/") for name in manifest["artifacts"]))
            self.assertNotIn("src/prepare_data.py", manifest["code_and_environment"])


if __name__ == "__main__":
    unittest.main()
