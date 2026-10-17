"""Contract and repeatability tests for the fictional public demonstration."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.audit_data import SOURCE_COLUMNS, load_table, sha256
from src.generate_demo_data import demo_frames, generate_demo_data
from src.prepare_data import prepare_frames


class DemoDataTests(unittest.TestCase):
    def test_repeatability_schema_and_synthetic_identifiers(self):
        a, b = demo_frames(customers_per_cohort=40, validation_customers=12)
        c, d = demo_frames(customers_per_cohort=40, validation_customers=12)
        pd.testing.assert_frame_equal(a, c)
        pd.testing.assert_frame_equal(b, d)
        self.assertEqual(list(a), [*SOURCE_COLUMNS, "noise"])
        self.assertEqual(list(b), list(SOURCE_COLUMNS))
        for frame in (a, b):
            for column in ("FK_DWH_CUSTOMERS", "FK_SUBSCRIPTIONS", "PK_INVOICES"):
                self.assertTrue(frame[column].str.startswith("SYN-").all())
        different, _ = demo_frames(seed=7, customers_per_cohort=40, validation_customers=12)
        self.assertFalse(a.equals(different))

    def test_both_classes_and_fallback_exist_in_each_role(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_demo_data(directory, customers_per_cohort=40, validation_customers=12)
            training = load_table(Path(directory) / "training.csv", training=True)
            scoring = load_table(Path(directory) / "validation.csv", training=False)
            tables, contract = prepare_frames(training, scoring)
            population = tables["training_metadata.csv"].merge(tables["training_targets.csv"], on=["fk_dwh_customers", "customer_start_date"])
            for _, group in population.groupby("partition"):
                self.assertEqual(set(group.loc[~group.fallback_required, "noise"]), {0, 1})
                self.assertGreater(int(group.fallback_required.sum()), 0)
            self.assertEqual(contract["identity_exclusions"]["union_training_customers"], 2)
            self.assertGreater(contract["submission"]["fallback_customers"], 0)
            self.assertGreater(tables["validation_metadata.csv"].groupby("fk_dwh_customers").size().max(), 1)

    def test_generated_files_repeat_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_demo_data(directory, customers_per_cohort=40, validation_customers=12)
            first = {p.name: sha256(p) for p in Path(directory).iterdir()}
            generate_demo_data(directory, customers_per_cohort=40, validation_customers=12)
            self.assertEqual(first, {p.name: sha256(p) for p in Path(directory).iterdir()})

    def test_original_or_unrecognized_inputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "training.csv"
            p.write_text("unrecognized source data")
            with self.assertRaisesRegex(ValueError, "Refusing"):
                generate_demo_data(directory)
            self.assertEqual(p.read_text(), "unrecognized source data")

    def test_invalid_sizes_and_seed_fail_before_writing(self):
        for kwargs in ({"seed": -1}, {"seed": True}, {"customers_per_cohort": 19}, {"validation_customers": 5}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    generate_demo_data(directory, **kwargs)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
