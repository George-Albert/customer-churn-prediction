"""Prepare customer/cohort features without fitting a churn model.

Row-level tables stay under ignored outputs/. IDs and cohort dates are linkage
keys, never predictors. Labels and partition metadata are exported separately.
"""

import argparse
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_data import CUSTOMER, load_table, parse_date, parse_numeric, sha256, write_report


KEYS = [CUSTOMER, "customer_start_date"]
ATTRIBUTES = ["flg_dwh_customer_is_business", "origin_group", "origin_subgroup"]
PARTITIONS = {
    "20230101": "learning", "20230201": "learning",
    "20230301": "learning", "20230401": "learning",
    "20230501": "development", "20230601": "evaluation",
}


def validate_source(frame, *, training):
    """Fail on unsupported tokens rather than silently summing partial records."""
    start = parse_date(frame.customer_start_date)
    invoice = parse_date(frame.fk_date_invoice)
    if start.isna().any() or start.dt.day.ne(1).any() or invoice.isna().any():
        raise ValueError("Invalid dates or customer start not at the first day of a month")
    if parse_numeric(frame.tenure).isna().any() or parse_numeric(frame.mrr).isna().any():
        raise ValueError("Invalid or nonfinite TENURE/MRR tokens")
    if not frame.flg_dwh_customer_is_business.isin(["0", "1"]).all():
        raise ValueError("Invalid business flag")
    for column in [CUSTOMER, "pk_invoices", "fk_subscriptions", "product_group", "product_subcategory"]:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Blank required field: {column}")
    if training and not frame.noise.isin(["True", "False"]).all():
        raise ValueError("Invalid supplied noise label")


def select_records(frame, *, training, mixed_ids=(), shared_ids=()):
    """Apply approved exclusions, retaining original zero-based row positions."""
    working = frame.reset_index(drop=True).copy()
    tenure = parse_numeric(working.tenure)
    start = parse_date(working.customer_start_date)
    invoice = parse_date(working.fk_date_invoice)
    in_window = invoice.ge(start) & invoice.lt(start + pd.DateOffset(months=2))
    reason = pd.Series("retained", index=working.index, dtype="string")
    reason.loc[~tenure.isin([0, 1])] = "outside_tenure_0_1"
    reason.loc[tenure.lt(0)] = "negative_tenure"
    reason.loc[tenure.isin([0, 1]) & ~in_window] = "checkout_outside_window"
    # Identity exclusions take precedence over row exclusions in the audit.
    if training:
        reason.loc[working[CUSTOMER].isin(shared_ids)] = "shared_training_customer"
        reason.loc[working[CUSTOMER].isin(mixed_ids)] = "mixed_cohort_training_customer"
    audit = working[KEYS].copy()
    audit.insert(0, "input_row", working.index)
    audit["disposition"] = reason
    usable = working.loc[reason.eq("retained")].copy()
    usable["tenure"] = tenure.loc[usable.index].astype(int)
    usable["mrr"] = parse_numeric(usable.mrr)
    excluded = set(mixed_ids) | set(shared_ids) if training else set()
    registry = working.loc[~working[CUSTOMER].isin(excluded), KEYS].drop_duplicates()
    registry = registry.sort_values(KEYS).set_index(KEYS)
    registry["usable_rows"] = usable.groupby(KEYS).size().reindex(registry.index, fill_value=0)
    registry["fallback_required"] = registry.usable_rows.eq(0)
    if training:
        partition = registry.index.get_level_values("customer_start_date").map(PARTITIONS)
        if partition.isna().any():
            raise ValueError("Training cohort outside the approved January-June 2023 partitions")
        registry["partition"] = partition
    else:
        registry["partition"] = "submission"
    return usable, registry, audit


def build_features(usable, vocabulary):
    """Aggregate records; a missing month remains missing, including its sums."""
    grouped = usable.groupby(KEYS, sort=True)
    if grouped[ATTRIBUTES].nunique().gt(1).any().any():
        raise ValueError("Conflicting usable customer attributes; no arbitrary value selected")
    features = grouped[ATTRIBUTES].first().rename(columns={"flg_dwh_customer_is_business": "is_business"})
    features["is_business"] = features.is_business.astype(int)
    subscriptions = []
    for month in [0, 1]:
        rows = usable.loc[usable.tenure.eq(month)].copy()
        # An observed month with no entry of a sign has a zero signed subtotal.
        rows["positive_mrr"] = rows.mrr.clip(lower=0)
        rows["negative_mrr"] = rows.mrr.clip(upper=0)
        monthly = rows.groupby(KEYS).agg(
            n_records=("pk_invoices", "size"),
            n_invoices=("pk_invoices", "nunique"),
            n_subscriptions=("fk_subscriptions", "nunique"),
            n_product_groups=("product_group", "nunique"),
            invoice_record_mrr_sum=("mrr", "sum"),
            invoice_record_positive_mrr_sum=("positive_mrr", "sum"),
            invoice_record_negative_mrr_sum=("negative_mrr", "sum"),
        ).add_suffix(f"_month_{month}")
        features[f"month_{month}_observed"] = features.index.isin(monthly.index).astype(int)
        features = features.join(monthly)
        subscriptions.append(rows.groupby(KEYS).fk_subscriptions.agg(frozenset))
    features["n_product_subcategories_total"] = grouped.product_subcategory.nunique()
    product_sets = grouped.product_subcategory.agg(frozenset)
    for column, category in vocabulary.items():
        features[column] = product_sets.map(lambda values: int(category in values))
    features["mrr_change_month_1_minus_0"] = (
        features.invoice_record_mrr_sum_month_1 - features.invoice_record_mrr_sum_month_0
    )
    features["subscription_count_change_month_1_minus_0"] = (
        features.n_subscriptions_month_1 - features.n_subscriptions_month_0
    )
    both = features.month_0_observed.eq(1) & features.month_1_observed.eq(1)
    features["n_subscriptions_added_month_1"] = np.nan
    features["n_subscriptions_absent_month_1"] = np.nan
    common = features.index[both]
    for column, left, right in (
        ("n_subscriptions_added_month_1", 1, 0),
        ("n_subscriptions_absent_month_1", 0, 1),
    ):
        features.loc[common, column] = [
            len(a - b) for a, b in zip(subscriptions[left].reindex(common), subscriptions[right].reindex(common))
        ]
    return features.sort_index()


def prepare_frames(train, submission, *, vocabulary_partitions=("learning",)):
    """Learn vocabulary from explicit labeled partitions; default excludes May/June."""
    if not vocabulary_partitions or not set(vocabulary_partitions).issubset(set(PARTITIONS.values())):
        raise ValueError("Invalid vocabulary partitions")
    validate_source(train, training=True)
    validate_source(submission, training=False)
    mixed_ids = set(train.groupby(CUSTOMER).customer_start_date.nunique().loc[lambda n: n.gt(1)].index)
    shared_ids = set(train[CUSTOMER]) & set(submission[CUSTOMER])
    usable, registry, row_audit = select_records(
        train, training=True, mixed_ids=mixed_ids, shared_ids=shared_ids,
    )
    validation_usable, validation_registry, validation_audit = select_records(submission, training=False)
    eligible = train.loc[~train[CUSTOMER].isin(mixed_ids | shared_ids)]
    if eligible.groupby(CUSTOMER).noise.nunique().gt(1).any():
        raise ValueError("Conflicting eligible noise labels; no majority correction allowed")
    targets = eligible.groupby(KEYS).noise.first().map({"True": 1, "False": 0}).to_frame("noise")
    learning_keys = registry.index[registry.partition.isin(vocabulary_partitions) & ~registry.fallback_required]
    learning_rows = usable.loc[pd.MultiIndex.from_frame(usable[KEYS]).isin(learning_keys)]
    if learning_rows.empty:
        raise ValueError("No usable learning records for the feature vocabulary")
    vocabulary = {
        f"has_product_subcategory_{index:03d}": category
        for index, category in enumerate(sorted(learning_rows.product_subcategory.unique()))
    }
    train_features = build_features(usable, vocabulary)
    validation_features = build_features(validation_usable, vocabulary)
    for rows, metadata in ((usable, registry), (validation_usable, validation_registry)):
        unknown = rows.loc[~rows.product_subcategory.isin(vocabulary.values())]
        metadata["unseen_product_records"] = unknown.groupby(KEYS).size().reindex(metadata.index, fill_value=0)
    submission_ids = pd.DataFrame({CUSTOMER: sorted(submission[CUSTOMER].unique())})
    if set(validation_registry.index.get_level_values(CUSTOMER)) != set(submission_ids[CUSTOMER]):
        raise ValueError("Submission ID coverage changed")
    if not registry.index.equals(targets.index) or not registry.index.is_unique:
        raise ValueError("Training metadata/target linkage is inconsistent")
    if not train_features.columns.equals(validation_features.columns):
        raise ValueError("Training and submission feature schemas differ")
    tables = {
        "training_features.csv": train_features.reset_index(),
        "validation_features.csv": validation_features.reset_index(),
        "training_metadata.csv": registry.reset_index(),
        "validation_metadata.csv": validation_registry.reset_index(),
        "training_targets.csv": targets.reset_index(),
        "submission_ids.csv": submission_ids,
        "training_row_audit.csv": row_audit,
        "validation_row_audit.csv": validation_audit,
    }
    partitions = []
    for name in ["learning", "development", "evaluation"]:
        group = registry.loc[registry.partition.eq(name)]
        partitions.append({"partition": name, "customers": len(group),
                           "usable_customers": int((~group.fallback_required).sum()),
                           "fallback_customers": int(group.fallback_required.sum())})
    usable_ids = validation_features.index.get_level_values(CUSTOMER).unique()
    summary = {
        "phase": 2,
        "vocabulary_partitions": list(vocabulary_partitions),
        "identity_exclusions": {"mixed_cohort_training_customers": len(mixed_ids),
                                "shared_training_customers": len(shared_ids),
                                "union_training_customers": len(mixed_ids | shared_ids)},
        "row_dispositions": {
            name: {str(key): int(value) for key, value in audit.disposition.value_counts().sort_index().items()}
            for name, audit in (("training", row_audit), ("validation", validation_audit))
        },
        "partitions": partitions,
        "submission": {"customers": len(submission_ids), "cohort_groups": len(validation_registry),
                       "usable_customers": len(usable_ids),
                       "fallback_customers": len(submission_ids) - len(usable_ids),
                       "usable_cohort_groups": len(validation_features)},
        "predictor_columns": list(train_features.columns),
        "product_vocabulary": vocabulary,
        "unseen_product_records": {"training": int(registry.unseen_product_records.sum()),
                                   "validation": int(validation_registry.unseen_product_records.sum())},
        "limitations": [
            "Supplied noise construction and historical label/feature availability are unverified.",
            "Retrospective cohort evaluation does not establish annual seasonality or a deployable backtest.",
            "MRR fields retain source scale/sign and are not an audited revenue KPI.",
            "Fallback probabilities and ambiguous-ID probability averaging belong to model scoring, not preparation.",
            "No model, imputation, scaling, balancing or TENURE-only sensitivity is fitted in this phase.",
        ],
    }
    return tables, summary


def generate_preparation(input_dir, output_dir):
    """Write deterministic local tables and a separate phase-2 provenance manifest."""
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    project_root = Path(__file__).resolve().parents[1]
    # Individual customer tables must stay in the repository's ignored output tree.
    if not output_dir.is_relative_to(project_root / "outputs") or output_dir.is_relative_to(input_dir):
        raise ValueError("Preparation output must be under project outputs/ and outside input/")
    inputs = [input_dir / name for name in ("case_study.ipynb", "training.csv", "validation.csv")]
    input_hashes = {path.name: sha256(path) for path in inputs}
    train = load_table(input_dir / "training.csv", training=True)
    submission = load_table(input_dir / "validation.csv", training=False)
    tables, summary = prepare_frames(train, submission)
    if input_hashes != {path.name: sha256(path) for path in inputs}:
        raise ValueError("An input changed during preparation")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, na_rep="", lineterminator="\n")
    write_report(summary, output_dir / "preparation_summary.json")
    sources = [project_root / path for path in (
        "src/audit_data.py", "src/prepare_data.py", "tests/test_prepare_data.py",
        "notebooks/customer_churn_demo.ipynb", "requirements.txt", "doc/data_dictionary.md", "doc/data_quality.md",
    )]
    manifest = {
        "phase": 2, "seed": None, "seed_note": "Deterministic aggregation; no model fitting.",
        "inputs": input_hashes,
        "code_and_contract": {str(path.relative_to(project_root)): sha256(path) for path in sources},
        "python": platform.python_version(),
        "packages": {package: importlib.metadata.version(package) for package in ("pandas", "numpy")},
        "artifacts": {name: sha256(output_dir / name) for name in [*tables, "preparation_summary.json"]},
        "reproduce": "python -m src.prepare_data --input-dir input --output-dir outputs/preparation",
        "csv_reading": "dtype='string', keep_default_na=False; parse predictors explicitly; blank numerics mean missing",
    }
    write_report(manifest, output_dir / "preparation_manifest.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/preparation"))
    args = parser.parse_args()
    summary = generate_preparation(args.input_dir, args.output_dir)
    print(f"Phase-2 preparation written to {args.output_dir}; {len(summary['predictor_columns'])} predictors; no fitted model")


if __name__ == "__main__":
    main()
