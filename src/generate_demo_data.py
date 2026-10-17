"""Generate fictional invoice histories without reading any source customer data.

The latent risk affects both observed activity and a noisy synthetic binary
outcome. These invented relationships demonstrate mechanics, not churn science.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_data import SOURCE_COLUMNS


SEED = 42


def demo_frames(seed=SEED, customers_per_cohort=200, validation_customers=240):
    """Return source-shaped synthetic frames; no real data are used as a template."""
    if (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or
            not isinstance(customers_per_cohort, int) or isinstance(customers_per_cohort, bool) or
            not isinstance(validation_customers, int) or isinstance(validation_customers, bool) or
            customers_per_cohort < 20 or validation_customers < 6):
        raise ValueError("Use a nonnegative integer seed, at least 20 customers/cohort and 6 validation customers")
    rng = np.random.default_rng(seed)
    products = (("Hosting", "Shared Hosting"), ("Mail", "E-mail"), ("Domain", "Domain Names"))

    def history(customer, cohort, index, *, labeled, fallback=False, forced_label=None):
        start = pd.Timestamp(cohort)
        risk = float(rng.uniform(.05, .95))
        label = bool(rng.random() < .18 + .60 * risk)
        if forced_label is not None:
            label = forced_label
        channel = ("NA", "NA") if index % 7 == 0 else (("Organic", "Search") if index % 2 else ("Paid", "Campaign"))
        if not labeled and index % 11 == 0:
            channel = ("New channel", "New source")
        business = str(index % 2)
        months = [0] if fallback or rng.random() < .12 + .38 * risk else [0, 1]
        rows = []
        for month in months:
            invoice_date = start + pd.DateOffset(months=6 if fallback else month, days=10)
            invoice = f"SYN-INV-{customer}-{month}"
            for product_index in range(2 if index % 3 == 0 else 1):
                group, category = products[(index + product_index) % len(products)]
                amount = float(np.exp(rng.normal(1.5 - .8 * risk, .45)))
                if index % 19 == 0:
                    amount = -amount / 8  # Fictional adjustment, not an asserted accounting interpretation.
                if index % 31 == 0:
                    amount = .0001
                rows.append(dict(zip(SOURCE_COLUMNS, (
                    invoice, invoice_date.strftime("%Y%m%d"), start.strftime("%Y%m%d"), str(month),
                    business, customer, f"SYN-SUB-{customer}-{product_index}", group, category,
                    f"{amount:.6f}", channel[1], channel[0],
                ))))
                if labeled:
                    rows[-1]["noise"] = str(label)
        return rows

    training = []
    fallback_count = max(2, customers_per_cohort // 10)
    for month in range(1, 7):
        cohort = f"2023-{month:02d}-01"
        for index in range(customers_per_cohort):
            fallback = index < fallback_count
            forced = index % 5 == 0 if fallback else None
            # Ensure both classes exist in every usable cohort, including small test fixtures.
            if index in (fallback_count, fallback_count + 1):
                forced = index == fallback_count + 1
            training.extend(history(f"SYN-TRAIN-{month:02d}-{index:05d}", cohort, index,
                                    labeled=True, fallback=fallback, forced_label=forced))
    validation = []
    for index in range(validation_customers):
        validation.extend(history(f"SYN-SCORE-{index:05d}", f"2024-{index % 3 + 1:02d}-01", index,
                                  labeled=False, fallback=index % 10 == 0))
    # Exercise ambiguity/overlap policy with fictional IDs; these are not copied source anomalies.
    training += history("SYN-MIXED-LABELED", "2023-01-01", 1, labeled=True, forced_label=False)
    training += history("SYN-MIXED-LABELED", "2023-05-01", 2, labeled=True, forced_label=True)
    training += history("SYN-SHARED", "2023-01-01", 3, labeled=True)
    validation += history("SYN-SHARED", "2024-01-01", 3, labeled=False)
    validation += history("SYN-SCORE-00001", "2024-03-01", 2, labeled=False)
    # A separate negative-TENURE observation must not invalidate the customer's valid rows.
    extra = next(row.copy() for row in training if row["FK_DWH_CUSTOMERS"] == f"SYN-TRAIN-01-{fallback_count:05d}")
    extra.update(TENURE="-1", FK_DATE_INVOICE="20221220", PK_INVOICES="SYN-NEGATIVE-TENURE")
    training.append(extra)
    return (pd.DataFrame(training, columns=[*SOURCE_COLUMNS, "noise"], dtype="string"),
            pd.DataFrame(validation, columns=SOURCE_COLUMNS, dtype="string"))


def generate_demo_data(output_dir=Path("input"), *, seed=SEED, customers_per_cohort=200, validation_customers=240):
    """Write deterministic inputs; refuse to replace an unidentified existing dataset."""
    output_dir = Path(output_dir)
    paths = [output_dir / name for name in ("training.csv", "validation.csv", "case_study.ipynb")]
    if any(path.exists() for path in paths):
        try:
            recognized = json.loads(paths[2].read_text()).get("metadata", {}).get("synthetic_demo") is True
        except (OSError, ValueError):
            recognized = False
        if not recognized:
            raise ValueError("Refusing to overwrite inputs not marked as generated synthetic data")
    training, validation = demo_frames(seed, customers_per_cohort, validation_customers)
    brief = {
        "cells": [{"cell_type": "markdown", "id": "synthetic-demo-brief", "metadata": {},
                   "source": ["# Synthetic customer churn demonstration\n", "All records, identifiers and outcomes are fictional.\n",
                              "The binary noise target is generated for a fixed-horizon mechanics demonstration.\n",
                              "It is not observed customer churn and has no business-performance interpretation.\n"]}],
        "metadata": {"synthetic_demo": True, "seed": seed}, "nbformat": 4, "nbformat_minor": 5,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(paths[0], index=False, lineterminator="\n")
    validation.to_csv(paths[1], index=False, lineterminator="\n")
    paths[2].write_text(json.dumps(brief, indent=2) + "\n")
    return {"data_kind": "synthetic", "seed": seed, "training_rows": len(training),
            "training_customers": int(training.FK_DWH_CUSTOMERS.nunique()),
            "validation_rows": len(validation), "validation_customers": int(validation.FK_DWH_CUSTOMERS.nunique())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("input"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--customers-per-cohort", type=int, default=200)
    parser.add_argument("--validation-customers", type=int, default=240)
    args = parser.parse_args()
    print(json.dumps(generate_demo_data(args.output_dir, seed=args.seed,
                                       customers_per_cohort=args.customers_per_cohort,
                                       validation_customers=args.validation_customers), indent=2))


if __name__ == "__main__":
    main()
