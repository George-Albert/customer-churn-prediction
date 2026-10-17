"""Synthetic cases protect the audit's handling of ambiguous source records."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.audit_data import COLUMN_MAP, SOURCE_COLUMNS, build_report, dataset_summary, load_table, parse_date, sha256


def record(**changes):
    row = dict(zip(SOURCE_COLUMNS, (
        "invoice-1", "20230115", "20230101", "0", "1", "customer-1", "subscription-1",
        "Hosting", "Shared Hosting", "10.50", "NA", "NA",
    )))
    row["noise"] = "True"
    row.update(changes)
    return row


def normalized(rows):
    return pd.DataFrame(rows, dtype="string").rename(columns=COLUMN_MAP)


class AuditTests(unittest.TestCase):
    def test_both_datasets_share_lowercase_predictor_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "training.csv"
            validation_path = Path(directory) / "validation.csv"
            pd.DataFrame([record()]).to_csv(training_path, index=False)
            pd.DataFrame([record()]).drop(columns="noise").to_csv(validation_path, index=False)
            before = [sha256(path) for path in (training_path, validation_path)]
            training = load_table(training_path, training=True)
            validation = load_table(validation_path, training=False)
            self.assertEqual(list(training.drop(columns="noise")), list(validation))
            self.assertEqual(list(validation), [name.lower() for name in SOURCE_COLUMNS])
            self.assertTrue(validation.columns.is_unique)
            self.assertEqual(before, [sha256(path) for path in (training_path, validation_path)])

    def test_load_preserves_identifier_tokens_literal_na_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            pd.DataFrame([record(FK_DWH_CUSTOMERS="000123", ORIGIN_GROUP="")]).to_csv(path, index=False)
            before = sha256(path)
            frame = load_table(path, training=True)
            self.assertEqual(frame.fk_dwh_customers.iloc[0], "000123")
            self.assertEqual(frame.origin_subgroup.iloc[0], "NA")
            self.assertEqual(frame.origin_group.iloc[0], "")
            self.assertEqual(sha256(path), before)

    def test_missing_required_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            pd.DataFrame([record()]).drop(columns="TENURE").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing=.*TENURE"):
                load_table(path, training=True)

    def test_validation_rejects_unexpected_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            pd.DataFrame([record()]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "extra=.*noise"):
                load_table(path, training=False)

    def test_blank_customer_id_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            pd.DataFrame([record(FK_DWH_CUSTOMERS=" ")]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Blank customer"):
                load_table(path, training=True)

    def test_empty_dataset_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            pd.DataFrame(columns=[*SOURCE_COLUMNS, "noise"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Empty dataset"):
                load_table(path, training=True)

    def test_dates_require_eight_digits_and_a_real_calendar_date(self):
        values = pd.Series(["20230228", "20230230", "2023011", "not-a-date"], dtype="string")
        parsed = parse_date(values)
        self.assertEqual(parsed.notna().tolist(), [True, False, False, False])

    def test_date_disagreement_does_not_rewrite_tenure(self):
        frame = normalized([record(TENURE="1", FK_DATE_INVOICE="20230115")])
        original = frame.copy(deep=True)
        result = dataset_summary(frame)
        self.assertEqual(result["dates"]["tenure_month_delta_mismatch_rows"], 1)
        self.assertEqual(result["tenure"]["eligible_rows"], 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_invalid_dates_are_not_counted_as_date_disagreement(self):
        result = dataset_summary(normalized([record(FK_DATE_INVOICE="invalid")]))
        self.assertEqual(result["dates"]["invalid_invoice_rows"], 1)
        self.assertEqual(result["dates"]["comparable_rows"], 0)
        self.assertEqual(result["dates"]["tenure_month_delta_mismatch_rows"], 0)

    def test_negative_tenure_and_negative_mrr_remain_in_raw_audit(self):
        result = dataset_summary(normalized([record(TENURE="-1", mrr="-7.25")]))
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["tenure"]["negative_rows"], 1)
        self.assertEqual(result["tenure"]["customers_without_month_0_1"], 1)
        self.assertEqual(result["mrr"]["negative_rows"], 1)
        self.assertEqual(result["numeric"]["mrr"]["minimum"], -7.25)

    def test_target_distribution_counts_customers_not_invoice_records(self):
        rows = [record(), record(PK_INVOICES="invoice-2"), record(FK_DWH_CUSTOMERS="customer-2", noise="False")]
        result = dataset_summary(normalized(rows))["target"]
        self.assertEqual(result["row_counts"], {"False": 1, "True": 2})
        self.assertEqual(result["consistent_valid_customer_counts"], {"False": 1, "True": 1})
        self.assertEqual(result["consistent_customer_true_fraction"], 0.5)

    def test_conflicting_customer_labels_are_not_resolved_by_majority(self):
        rows = [record(), record(PK_INVOICES="invoice-2"), record(noise="False")]
        result = dataset_summary(normalized(rows))["target"]
        self.assertEqual(result["conflicting_customers"], 1)
        self.assertEqual(result["consistent_valid_customer_counts"], {})
        self.assertIsNone(result["consistent_customer_true_fraction"])
        self.assertEqual(result["rows_for_conflicting_customers_in_months_0_1"], 3)

    def test_invalid_target_is_flagged_and_excluded_from_distribution(self):
        result = dataset_summary(normalized([record(noise="maybe")]))["target"]
        self.assertEqual(result["invalid_rows"], 1)
        self.assertEqual(result["excluded_from_customer_distribution"], 1)

    def test_nonfinite_numeric_tokens_are_flagged_without_entering_summary(self):
        result = dataset_summary(normalized([record(mrr="inf"), record(mrr="bad", TENURE="bad")]))
        self.assertEqual(result["numeric"]["mrr"]["invalid_or_blank_rows"], 2)
        self.assertIsNone(result["numeric"]["mrr"]["maximum"])
        self.assertEqual(result["numeric"]["tenure"]["invalid_or_blank_rows"], 1)

    def test_equivalent_numeric_tenure_representation_counts_month_presence(self):
        result = dataset_summary(normalized([record(TENURE="0.0")]))
        self.assertEqual(result["tenure"]["customers_with_month_0"], 1)

    def test_conflicting_customer_attributes_are_reported(self):
        rows = [record(), record(CUSTOMER_START_DATE="20230201", FLG_DWH_CUSTOMER_IS_BUSINESS="0")]
        result = dataset_summary(normalized(rows))
        self.assertEqual(result["customers_with_multiple_values"]["customer_start_date"], 1)
        self.assertEqual(result["customers_with_multiple_values"]["flg_dwh_customer_is_business"], 1)
        self.assertEqual(sum(result["cohort_customer_memberships"].values()), 2)

    def test_customer_type_distribution_does_not_use_an_arbitrary_conflicting_row(self):
        rows = [record(), record(FLG_DWH_CUSTOMER_IS_BUSINESS="0"),
                record(FK_DWH_CUSTOMERS="customer-2", FLG_DWH_CUSTOMER_IS_BUSINESS="0")]
        result = dataset_summary(normalized(rows))["business_flag"]
        self.assertEqual(result["consistent_valid_customer_counts"], {"0": 1})
        self.assertEqual(result["excluded_from_customer_distribution"], 1)

    def test_shared_invoice_is_not_an_exact_duplicate(self):
        rows = [record(), record(FK_SUBSCRIPTIONS="subscription-2", PRODUCT_GROUP="Domain")]
        result = dataset_summary(normalized(rows))
        self.assertEqual(result["invoices"], 1)
        self.assertEqual(result["exact_duplicate_rows"], 0)

    def test_exact_duplicate_is_reported_without_deletion(self):
        result = dataset_summary(normalized([record(), record()]))
        self.assertEqual(result["exact_duplicate_rows"], 1)
        self.assertEqual(result["rows"], 2)

    def test_cross_dataset_overlap_and_unknown_categories_are_aggregate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([record()]).to_csv(root / "training.csv", index=False)
            pd.DataFrame([record(PRODUCT_GROUP="New Product")]).drop(columns="noise").to_csv(root / "validation.csv", index=False)
            result = build_report(root)
            self.assertEqual(result["cross_dataset"]["shared_identifiers"]["fk_dwh_customers"], 1)
            self.assertEqual(result["cross_dataset"]["validation_unseen_categories"]["product_group"], ["New Product"])
            self.assertNotIn("customer-1", str(result))


if __name__ == "__main__":
    unittest.main()
