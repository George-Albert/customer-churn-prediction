"""Synthetic preparation cases protect temporal eligibility and data separation."""

import unittest

import pandas as pd

from src.prepare_data import KEYS, generate_preparation, prepare_frames, select_records
from tests.test_audit_data import normalized, record


def submission(rows=None):
    if rows is None:
        rows = [record(FK_DWH_CUSTOMERS="validation-1", CUSTOMER_START_DATE="20240101", FK_DATE_INVOICE="20240115")]
    return normalized(rows).drop(columns="noise")


class PreparationTests(unittest.TestCase):
    def test_calendar_boundaries_and_negative_history_preserve_source(self):
        frame = normalized([
            record(FK_DATE_INVOICE="20221231"), record(FK_DATE_INVOICE="20230101"),
            record(TENURE="1", FK_DATE_INVOICE="20230228"),
            record(TENURE="1", FK_DATE_INVOICE="20230301"), record(TENURE="-1"),
        ])
        original = frame.copy(deep=True)
        usable, metadata, audit = select_records(frame, training=True)
        self.assertEqual(len(usable), 2)
        self.assertEqual(audit.disposition.tolist(), ["checkout_outside_window", "retained", "retained",
                                                     "checkout_outside_window", "negative_tenure"])
        self.assertEqual(audit.input_row.tolist(), list(range(5)))
        self.assertFalse(metadata.fallback_required.iloc[0])
        pd.testing.assert_frame_equal(frame, original)

    def test_calendar_window_handles_year_boundary_and_leap_day(self):
        frame = submission([
            record(CUSTOMER_START_DATE="20231201", FK_DATE_INVOICE="20240131", TENURE="1"),
            record(CUSTOMER_START_DATE="20231201", FK_DATE_INVOICE="20240201", TENURE="1"),
            record(FK_DWH_CUSTOMERS="leap", CUSTOMER_START_DATE="20240201", FK_DATE_INVOICE="20240229"),
        ])
        usable, _, _ = select_records(frame, training=False)
        self.assertEqual(len(usable), 2)

    def test_invoice_date_does_not_reassign_tenure(self):
        tables, _ = prepare_frames(normalized([record(TENURE="1", FK_DATE_INVOICE="20230115")]), submission())
        row = tables["training_features.csv"].iloc[0]
        self.assertEqual(row.month_0_observed, 0)
        self.assertEqual(row.month_1_observed, 1)

    def test_shared_invoice_product_rows_and_stable_subscriptions(self):
        rows = [record(), record(FK_SUBSCRIPTIONS="B", mrr="2"),
                record(TENURE="1", FK_DATE_INVOICE="20230215", mrr="3")]
        tables, _ = prepare_frames(normalized(rows), submission())
        row = tables["training_features.csv"].iloc[0]
        self.assertEqual(row.n_records_month_0, 2)
        self.assertEqual(row.n_invoices_month_0, 1)
        self.assertEqual(row.n_subscriptions_month_0, 2)
        self.assertEqual(row.n_subscriptions_month_1, 1)
        self.assertEqual(row.n_subscriptions_added_month_1, 0)
        self.assertEqual(row.n_subscriptions_absent_month_1, 1)

    def test_signed_tiny_and_cancelled_mrr_are_not_clipped(self):
        rows = [record(mrr="5"), record(mrr="-5"), record(TENURE="1", mrr="0.0009")]
        tables, _ = prepare_frames(normalized(rows), submission())
        row = tables["training_features.csv"].iloc[0]
        self.assertEqual(row.invoice_record_mrr_sum_month_0, 0)
        self.assertEqual(row.invoice_record_positive_mrr_sum_month_0, 5)
        self.assertEqual(row.invoice_record_negative_mrr_sum_month_0, -5)
        self.assertAlmostEqual(row.invoice_record_mrr_sum_month_1, 0.0009)
        self.assertAlmostEqual(row.mrr_change_month_1_minus_0, 0.0009)

    def test_missing_month_and_literal_na_are_distinct_from_zero(self):
        tables, _ = prepare_frames(normalized([record(mrr="0")]), submission())
        row = tables["training_features.csv"].iloc[0]
        self.assertEqual(row.origin_group, "NA")
        self.assertEqual(row.invoice_record_mrr_sum_month_0, 0)
        self.assertTrue(pd.isna(row.invoice_record_mrr_sum_month_1))
        self.assertTrue(pd.isna(row.invoice_record_positive_mrr_sum_month_1))
        self.assertTrue(pd.isna(row.n_invoices_month_1))
        self.assertTrue(pd.isna(row.n_subscriptions_added_month_1))
        self.assertTrue(pd.isna(row.mrr_change_month_1_minus_0))

    def test_blank_acquisition_token_is_not_literal_na(self):
        tables, _ = prepare_frames(normalized([record(ORIGIN_GROUP="", ORIGIN_SUBGROUP="")]), submission())
        self.assertEqual(tables["training_features.csv"].origin_group.iloc[0], "")

    def test_mixed_cohort_and_shared_training_ids_are_excluded(self):
        rows = [record(), record(FK_DWH_CUSTOMERS="mixed"),
                record(FK_DWH_CUSTOMERS="mixed", CUSTOMER_START_DATE="20230201", FK_DATE_INVOICE="20230215", noise="False"),
                record(FK_DWH_CUSTOMERS="shared")]
        validation = submission([record(FK_DWH_CUSTOMERS="shared", CUSTOMER_START_DATE="20240101", FK_DATE_INVOICE="20240115")])
        tables, summary = prepare_frames(normalized(rows), validation)
        self.assertEqual(tables["training_targets.csv"].fk_dwh_customers.tolist(), ["customer-1"])
        self.assertEqual(summary["identity_exclusions"]["union_training_customers"], 2)
        self.assertEqual(tables["submission_ids.csv"].fk_dwh_customers.tolist(), ["shared"])

    def test_fallback_customers_keep_labels_and_submission_coverage(self):
        rows = [record(), record(FK_DWH_CUSTOMERS="fallback", FK_DATE_INVOICE="20230901", noise="False")]
        validation = submission([record(FK_DWH_CUSTOMERS="validation-1", FK_DATE_INVOICE="20240901", CUSTOMER_START_DATE="20240101")])
        tables, summary = prepare_frames(normalized(rows), validation)
        metadata = tables["training_metadata.csv"].set_index("fk_dwh_customers")
        self.assertTrue(metadata.loc["fallback", "fallback_required"])
        self.assertEqual(len(tables["training_targets.csv"]), 2)
        self.assertEqual(len(tables["training_features.csv"]), 1)
        self.assertEqual(len(tables["validation_features.csv"]), 0)
        self.assertEqual(summary["submission"]["fallback_customers"], 1)

    def test_ambiguous_submission_groups_remain_separate(self):
        validation = submission([
            record(FK_DWH_CUSTOMERS="validation-1", CUSTOMER_START_DATE="20240101", FK_DATE_INVOICE="20240115", mrr="2"),
            record(FK_DWH_CUSTOMERS="validation-1", CUSTOMER_START_DATE="20240201", FK_DATE_INVOICE="20240215", mrr="7", ORIGIN_GROUP="Organic"),
        ])
        tables, _ = prepare_frames(normalized([record()]), validation)
        self.assertEqual(len(tables["submission_ids.csv"]), 1)
        self.assertEqual(len(tables["validation_features.csv"]), 2)
        self.assertEqual(tables["validation_features.csv"].invoice_record_mrr_sum_month_0.tolist(), [2, 7])

    def test_partitions_and_vocabulary_use_learning_only(self):
        rows = [record(PRODUCT_SUBCATEGORY="Learning product"),
                record(FK_DWH_CUSTOMERS="dev", CUSTOMER_START_DATE="20230501", FK_DATE_INVOICE="20230515", PRODUCT_SUBCATEGORY="Dev product"),
                record(FK_DWH_CUSTOMERS="eval", CUSTOMER_START_DATE="20230601", FK_DATE_INVOICE="20230615", PRODUCT_SUBCATEGORY="Eval product")]
        tables, summary = prepare_frames(normalized(rows), submission())
        metadata = tables["training_metadata.csv"].set_index("fk_dwh_customers")
        self.assertEqual(metadata.partition.to_dict(), {"customer-1": "learning", "dev": "development", "eval": "evaluation"})
        self.assertEqual(list(summary["product_vocabulary"].values()), ["Learning product"])
        self.assertEqual(metadata.loc["dev", "unseen_product_records"], 1)
        self.assertNotIn("noise", summary["predictor_columns"])
        self.assertFalse(set(KEYS + ["partition", "fallback_required"]) & set(summary["predictor_columns"]))
        self.assertEqual(tables["training_features.csv"].columns.tolist(), tables["validation_features.csv"].columns.tolist())

    def test_labels_do_not_change_predictors_and_inputs_stay_immutable(self):
        frame, validation = normalized([record()]), submission()
        original, validation_original = frame.copy(deep=True), validation.copy(deep=True)
        before, _ = prepare_frames(frame, validation)
        changed = frame.assign(noise="False")
        after, _ = prepare_frames(changed, validation)
        pd.testing.assert_frame_equal(before["training_features.csv"], after["training_features.csv"])
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(validation, validation_original)

    def test_conflicting_labels_are_not_resolved_by_majority(self):
        with self.assertRaisesRegex(ValueError, "Conflicting eligible noise"):
            prepare_frames(normalized([record(), record(noise="False")]), submission())

    def test_conflicting_usable_attributes_are_not_selected_arbitrarily(self):
        with self.assertRaisesRegex(ValueError, "Conflicting usable customer attributes"):
            prepare_frames(normalized([record(), record(ORIGIN_GROUP="Organic")]), submission())

    def test_attributes_from_excluded_rows_do_not_supply_predictors(self):
        rows = [record(), record(TENURE="-1", ORIGIN_GROUP="Organic", FLG_DWH_CUSTOMER_IS_BUSINESS="0", mrr="999")]
        tables, _ = prepare_frames(normalized(rows), submission())
        row = tables["training_features.csv"].iloc[0]
        self.assertEqual(row.origin_group, "NA")
        self.assertEqual(row.is_business, 1)
        self.assertEqual(row.invoice_record_mrr_sum_month_0, 10.5)

    def test_customer_outputs_cannot_be_written_to_source_or_tracked_directories(self):
        for output in ["input", "doc", "src"]:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "under project outputs"):
                generate_preparation("input", output)

    def test_unsupported_cohort_is_not_silently_assigned(self):
        with self.assertRaisesRegex(ValueError, "outside the approved"):
            prepare_frames(normalized([record(CUSTOMER_START_DATE="20230701", FK_DATE_INVOICE="20230715")]), submission())

    def test_invalid_numeric_and_date_tokens_fail_before_aggregation(self):
        for changes in [{"mrr": "invalid"}, {"mrr": "inf"}, {"TENURE": "invalid"}, {"FK_DATE_INVOICE": "20230230"}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare_frames(normalized([record(**changes)]), submission())

    def test_row_order_does_not_change_features_or_vocabulary(self):
        frame = normalized([record(), record(TENURE="1", FK_SUBSCRIPTIONS="B", PRODUCT_SUBCATEGORY="Other", mrr="-2")])
        before, before_summary = prepare_frames(frame, submission())
        after, after_summary = prepare_frames(frame.iloc[::-1], submission())
        pd.testing.assert_frame_equal(before["training_features.csv"], after["training_features.csv"])
        self.assertEqual(before_summary["product_vocabulary"], after_summary["product_vocabulary"])


if __name__ == "__main__":
    unittest.main()
