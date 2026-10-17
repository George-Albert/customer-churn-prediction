"""Audit invoice records without correcting or discarding source observations.

All identifiers stay strings. Reports contain aggregate counts, never customer IDs.
The CLI expects the supplied training/validation schemas and fails on schema drift.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_COLUMNS = (
    "PK_INVOICES", "FK_DATE_INVOICE", "CUSTOMER_START_DATE", "TENURE",
    "FLG_DWH_CUSTOMER_IS_BUSINESS", "FK_DWH_CUSTOMERS", "FK_SUBSCRIPTIONS",
    "PRODUCT_GROUP", "PRODUCT_SUBCATEGORY", "mrr", "ORIGIN_SUBGROUP", "ORIGIN_GROUP",
)
COLUMN_MAP = {name: name.lower() for name in (*SOURCE_COLUMNS, "noise")}
CUSTOMER = "fk_dwh_customers"
CATEGORIES = ("product_group", "product_subcategory", "origin_subgroup", "origin_group")
CONSISTENCY_COLUMNS = (
    "customer_start_date", "flg_dwh_customer_is_business", "origin_group", "origin_subgroup",
)


def sha256(path):
    """Hash files in chunks so provenance does not require a second in-memory copy."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_table(path, *, training):
    """Keep source tokens, including literal NA, until their meaning is reviewed."""
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    expected = set(SOURCE_COLUMNS) | ({"noise"} if training else set())
    missing = sorted(expected - set(frame.columns))
    extra = sorted(set(frame.columns) - expected)
    if missing or extra:
        raise ValueError(f"Unexpected schema in {Path(path).name}: missing={missing}, extra={extra}")
    if frame.empty:
        raise ValueError(f"Empty dataset: {Path(path).name}")
    if frame["FK_DWH_CUSTOMERS"].str.strip().eq("").any():
        raise ValueError(f"Blank customer identifiers in {Path(path).name}")
    return frame.rename(columns=COLUMN_MAP)


def parse_date(series):
    """Parse only YYYYMMDD tokens; invalid dates remain visible as missing values."""
    eight_digits = series.str.fullmatch(r"\d{8}")
    return pd.to_datetime(series.where(eight_digits), format="%Y%m%d", errors="coerce")


def counts(series):
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def parse_numeric(series):
    """Non-numeric and infinite tokens are invalid diagnostics, never real amounts."""
    values = pd.to_numeric(series, errors="coerce").astype("Float64")
    return values.where(np.isfinite(values))


def numeric_summary(series):
    values = parse_numeric(series)
    valid = values.dropna()
    return {
        "invalid_or_blank_rows": int(values.isna().sum()),
        "minimum": float(valid.min()) if len(valid) else None,
        "median": float(valid.median()) if len(valid) else None,
        "maximum": float(valid.max()) if len(valid) else None,
    }


def dataset_summary(frame):
    """Describe all raw records; window counts are diagnostic, not cleaning rules."""
    tenure = parse_numeric(frame["tenure"])
    mrr = parse_numeric(frame["mrr"])
    invoice_date = parse_date(frame["fk_date_invoice"])
    start_date = parse_date(frame["customer_start_date"])
    month_delta = (invoice_date.dt.year - start_date.dt.year) * 12
    month_delta += invoice_date.dt.month - start_date.dt.month
    comparable = month_delta.notna() & tenure.notna()
    eligible = tenure.isin([0, 1])
    customers = frame[CUSTOMER].nunique()
    consistent = {
        column: int((frame.groupby(CUSTOMER)[column].nunique() > 1).sum())
        for column in CONSISTENCY_COLUMNS
    }
    cohort_pairs = frame[[CUSTOMER, "customer_start_date"]].drop_duplicates()
    monthly_presence = pd.DataFrame({CUSTOMER: frame[CUSTOMER], "tenure": tenure}).loc[eligible].drop_duplicates()
    flag_sets = frame.groupby(CUSTOMER).flg_dwh_customer_is_business.agg(frozenset)
    valid_flag = flag_sets.map(lambda values: len(values) == 1 and values <= {"0", "1"})
    customer_flags = flag_sets[valid_flag].map(lambda values: next(iter(values)))
    result = {
        "rows": len(frame), "customers": int(customers),
        "invoices": int(frame["pk_invoices"].nunique()),
        "subscriptions": int(frame["fk_subscriptions"].nunique()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "columns": list(frame.columns),
        "loaded_dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "blank_rows_by_column": {col: int(frame[col].str.strip().eq("").sum()) for col in frame},
        "literal_NA_rows_by_column": {col: int(frame[col].eq("NA").sum()) for col in frame},
        "unique_values_by_column": {col: int(frame[col].nunique()) for col in frame},
        "numeric": {col: numeric_summary(frame[col]) for col in ("tenure", "mrr")},
        "tenure": {
            "counts": counts(frame["tenure"]),
            "negative_rows": int(tenure.lt(0).sum()),
            "outside_months_0_1_rows": int((~eligible).sum()),
            "eligible_rows": int(eligible.sum()),
            "customers_without_month_0_1": int(customers - frame.loc[eligible, CUSTOMER].nunique()),
            "customers_with_month_0": int(monthly_presence.loc[monthly_presence.tenure.eq(0), CUSTOMER].nunique()),
            "customers_with_month_1": int(monthly_presence.loc[monthly_presence.tenure.eq(1), CUSTOMER].nunique()),
        },
        "dates": {
            "invalid_invoice_rows": int(invoice_date.isna().sum()),
            "invalid_start_rows": int(start_date.isna().sum()),
            "start_not_first_day_rows": int((start_date.notna() & start_date.dt.day.ne(1)).sum()),
            "invoice_min": invoice_date.min().strftime("%Y-%m-%d") if invoice_date.notna().any() else None,
            "invoice_max": invoice_date.max().strftime("%Y-%m-%d") if invoice_date.notna().any() else None,
            "comparable_rows": int(comparable.sum()),
            "tenure_month_delta_mismatch_rows": int((comparable & month_delta.ne(tenure)).sum()),
            "invoice_outside_months_0_1_rows": int((month_delta.notna() & ~month_delta.isin([0, 1])).sum()),
        },
        "mrr": {"negative_rows": int(mrr.lt(0).sum()), "zero_rows": int(mrr.eq(0).sum())},
        "business_flag_invalid_rows": int((~frame.flg_dwh_customer_is_business.isin(["0", "1"])).sum()),
        "business_flag": {
            "row_counts": counts(frame.flg_dwh_customer_is_business),
            "consistent_valid_customer_counts": counts(customer_flags),
            "excluded_from_customer_distribution": int((~valid_flag).sum()),
        },
        "customers_with_multiple_values": consistent,
        "cohort_customer_memberships": counts(cohort_pairs.customer_start_date),
        "cohort_note": "Unique customer/cohort pairs; inconsistent customers may belong to multiple cohorts.",
        "categories": {col: counts(frame[col]) for col in CATEGORIES},
        "subscriptions_linked_to_multiple_customers": int((frame.groupby("fk_subscriptions")[CUSTOMER].nunique() > 1).sum()),
    }
    if "noise" in frame:
        target_sets = frame.groupby(CUSTOMER).noise.agg(frozenset)
        valid_single = target_sets.map(lambda values: len(values) == 1 and values <= {"True", "False"})
        customer_labels = target_sets[valid_single].map(lambda values: next(iter(values)))
        window_sets = frame.loc[eligible].groupby(CUSTOMER).noise.agg(frozenset)
        window_conflicts = window_sets[window_sets.map(len).gt(1)].index
        result["target"] = {
            "row_counts": counts(frame.noise),
            "invalid_rows": int((~frame.noise.isin(["True", "False"])).sum()),
            "conflicting_customers": int(target_sets.map(len).gt(1).sum()),
            "consistent_valid_customer_counts": counts(customer_labels),
            "excluded_from_customer_distribution": int((~valid_single).sum()),
            "consistent_customer_true_fraction": float(customer_labels.eq("True").mean()) if len(customer_labels) else None,
            "conflicting_customers_in_months_0_1": len(window_conflicts),
            "rows_for_conflicting_customers_in_months_0_1": int((eligible & frame[CUSTOMER].isin(window_conflicts)).sum()),
        }
    return result


def build_report(input_dir, frames=None):
    """Reuse already loaded tables in the EDA; the standalone CLI loads them once."""
    input_dir = Path(input_dir)
    if frames is None:
        frames = {
            "training": load_table(input_dir / "training.csv", training=True),
            "validation": load_table(input_dir / "validation.csv", training=False),
        }
    train, validation = frames["training"], frames["validation"]
    return {
        "report_version": 1,
        "phase": "1: audit and descriptive EDA; no corrections or model fitting",
        "column_mapping": COLUMN_MAP,
        "datasets": {
            name: {"sha256": sha256(input_dir / name), "bytes": (input_dir / name).stat().st_size,
                   **dataset_summary(frame)}
            for name, frame in (("training.csv", train), ("validation.csv", validation))
        },
        "cross_dataset": {
            "shared_identifiers": {
                col: len(set(train[col]) & set(validation[col]))
                for col in (CUSTOMER, "fk_subscriptions", "pk_invoices")
            },
            "validation_unseen_categories": {
                col: sorted(set(validation[col]) - set(train[col])) for col in CATEGORIES
            },
        },
        "limitations": [
            "Literal NA is counted separately from blanks; its business meaning is unresolved.",
            "TENURE 0/1 defines an exploratory subset only, not an approved cleaning policy.",
            "Customer target proportions exclude contradictory or invalid labels; no label is repaired.",
            "Invoice dates and TENURE disagree; neither is silently used to overwrite the other.",
            "No customer identifiers or individual records are exported in this report.",
        ],
    }


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_quality_summary.json"))
    args = parser.parse_args()
    report = build_report(args.input_dir)
    write_report(report, args.output)
    print(f"Aggregate audit written to {args.output}")


if __name__ == "__main__":
    main()
