"""Create aggregate descriptive tables and figures without fitting transformations.

The provisional TENURE 0/1 subset is only used for customer-level descriptions.
Raw distributions, quality checks and cohort memberships retain every source row.
"""

import argparse
import importlib.metadata
import os
import platform
from pathlib import Path

# Keep library caches inside the project rather than the user's home directory.
_CACHE = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audit_data import CUSTOMER, build_report, load_table, parse_numeric, sha256, write_report


COLORS = {"training": "#286487", "validation": "#d17732"}


def customer_activity(frame):
    """Keep absent months missing; absence does not establish zero revenue or churn."""
    tenure = parse_numeric(frame.tenure)
    eligible = frame.loc[tenure.isin([0, 1])].copy()
    eligible["tenure"] = tenure.loc[eligible.index].astype(int).astype(str)
    eligible["mrr"] = parse_numeric(eligible.mrr)
    activity = eligible.groupby(CUSTOMER).agg(
        invoice_records=("pk_invoices", "size"),
        distinct_invoices=("pk_invoices", "nunique"),
        distinct_subscriptions=("fk_subscriptions", "nunique"),
        distinct_products=("product_group", "nunique"),
        observed_months=("tenure", "nunique"),
    )
    monthly = eligible.groupby([CUSTOMER, "tenure"]).mrr.sum(min_count=1).unstack("tenure")
    monthly = monthly.reindex(columns=["0", "1"]).rename(columns={
        "0": "invoice_record_mrr_sum_month_0", "1": "invoice_record_mrr_sum_month_1",
    })
    return activity.join(monthly)


def distribution_table(values, *, dataset, scope):
    """Export distribution statistics only, never customer-level feature tables."""
    rows = []
    for name, series in values.items():
        numeric = parse_numeric(series)
        stats = numeric.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append({"dataset": dataset, "scope": scope, "variable": name,
                     "missing": int(numeric.isna().sum()), **stats.to_dict()})
    return pd.DataFrame(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": "customer churn demo phase-1 EDA"})
    plt.close(fig)


def generate_eda(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    figures, tables = output_dir / "figures", output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    frames = {
        "training": load_table(input_dir / "training.csv", training=True),
        "validation": load_table(input_dir / "validation.csv", training=False),
    }
    report = build_report(input_dir, frames)
    write_report(report, output_dir / "data_quality_summary.json")
    activities = {name: customer_activity(frame) for name, frame in frames.items()}
    summaries = []
    cohort_rows, product_rows, acquisition_rows, monthly_rows = [], [], [], []
    for name, frame in frames.items():
        summaries.append(distribution_table({"mrr": frame.mrr}, dataset=name, scope="all invoice records"))
        summaries.append(distribution_table(activities[name], dataset=name, scope="customers with TENURE 0/1 records"))
        cohorts = frame[[CUSTOMER, "customer_start_date"]].drop_duplicates().groupby("customer_start_date").size()
        cohort_rows.extend({"dataset": name, "cohort": cohort, "customer_memberships": int(n)} for cohort, n in cohorts.items())
        for column, destination in (("product_group", product_rows), ("origin_group", acquisition_rows)):
            for category, n in frame[column].value_counts().sort_index().items():
                destination.append({"dataset": name, "category": category, "invoice_records": int(n),
                                    "fraction_of_invoice_records": float(n / len(frame))})
        tenure = parse_numeric(frame.tenure)
        eligible = frame.loc[tenure.isin([0, 1])].copy()
        eligible["tenure"] = tenure.loc[eligible.index].astype(int).astype(str)
        eligible["mrr"] = parse_numeric(eligible.mrr)
        for month, group in eligible.groupby("tenure"):
            monthly_rows.append({"dataset": name, "tenure": month, "invoice_records": len(group),
                                 "customers": int(group[CUSTOMER].nunique()),
                                 "distinct_invoices": int(group.pk_invoices.nunique()),
                                 "invoice_record_mrr_sum": float(group.mrr.sum(min_count=1))})
    pd.concat(summaries, ignore_index=True).to_csv(tables / "numeric_distributions.csv", index=False)
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(tables / "cohort_memberships.csv", index=False)
    products = pd.DataFrame(product_rows)
    products.to_csv(tables / "product_mix.csv", index=False)
    acquisitions = pd.DataFrame(acquisition_rows)
    acquisitions.to_csv(tables / "acquisition_mix.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(tables / "monthly_activity.csv", index=False)
    business_rows = []
    for name in frames:
        summary = report["datasets"][f"{name}.csv"]["business_flag"]
        for flag, n in summary["consistent_valid_customer_counts"].items():
            business_rows.append({"dataset": name, "is_business": flag, "customers": n,
                                  "excluded_inconsistent_or_invalid_customers": summary["excluded_from_customer_distribution"]})
    pd.DataFrame(business_rows).to_csv(tables / "customer_type_mix.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (name, frame) in zip(axes, frames.items()):
        axis.hist(parse_numeric(frame.mrr).dropna(), bins=100, color=COLORS[name])
        axis.set(title=f"{name.title()}: all invoice records", xlabel="MRR per record (unit unspecified)", ylabel="Invoice records")
        axis.set_yscale("log")
    fig.suptitle("Raw MRR distribution — full range, including negative values")
    save_figure(fig, figures / "raw_mrr.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, variable in zip(axes, ["invoice_records", "distinct_subscriptions"]):
        # Shared integer bins make the two customer distributions comparable.
        maximum = max(activity[variable].max() for activity in activities.values())
        bins = np.arange(0.5, maximum + 1.5, 1)
        for name, activity in activities.items():
            axis.hist(activity[variable], bins=bins, histtype="step", density=True,
                      label=name, color=COLORS[name], linewidth=1.5)
        axis.set(title=variable.replace("_", " ").title(), xlabel="Count per customer", ylabel="Fraction of customers")
        axis.set_yscale("log")
        axis.legend()
    fig.suptitle("Customer activity — exploratory TENURE 0/1 subset; full range")
    save_figure(fig, figures / "customer_activity.png")

    fig, axis = plt.subplots(figsize=(10, 4))
    for name in frames:
        subset = cohorts.loc[cohorts.dataset.eq(name)]
        axis.bar(subset.cohort, subset.customer_memberships, color=COLORS[name], label=name)
    axis.set(title="Raw cohort memberships — customers with conflicting cohorts can appear twice",
             xlabel="Customer start month (YYYYMMDD)", ylabel="Unique customer/cohort pairs")
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylim(0, cohorts.customer_memberships.max() * 1.25)
    axis.legend()
    save_figure(fig, figures / "cohorts.png")

    for table, filename, title in (
        (products, "product_mix.png", "Product mix"),
        (acquisitions, "acquisition_mix.png", "Acquisition channel mix"),
    ):
        pivot = table.pivot(index="category", columns="dataset", values="fraction_of_invoice_records").fillna(0)
        fig, axis = plt.subplots(figsize=(11, max(4, len(pivot) * 0.3)))
        pivot.plot.barh(ax=axis, color=[COLORS[col] for col in pivot.columns])
        axis.set(title=f"{title} — all invoice records", xlabel="Fraction of invoice records", ylabel="")
        save_figure(fig, figures / filename)

    target = report["datasets"]["training.csv"]["target"]
    label_counts = target["consistent_valid_customer_counts"]
    pd.DataFrame([{"noise": label, "customers": count} for label, count in sorted(label_counts.items())]).to_csv(
        tables / "target_distribution.csv", index=False,
    )
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(["Retained (False)", "Churn (True)"], [label_counts.get("False", 0), label_counts.get("True", 0)], color=["#47856c", "#b85b4a"])
    axis.set(title=f"Training target — consistent valid labels only\n{target['excluded_from_customer_distribution']} customer(s) excluded from this chart",
             ylabel="Customers")
    save_figure(fig, figures / "target_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (name, activity) in zip(axes, activities.items()):
        data = [activity[f"invoice_record_mrr_sum_month_{month}"].dropna() for month in [0, 1]]
        axis.boxplot(data, tick_labels=["Month 0", "Month 1"], showfliers=True)
        axis.set(title=name.title(), ylabel="Sum of MRR fields per customer/month (symlog)")
        axis.set_yscale("symlog", linthresh=1)
    fig.suptitle("Exploratory TENURE 0/1 subset — absent months remain missing; all outliers shown")
    save_figure(fig, figures / "monthly_record_mrr.png")

    project_root = Path(__file__).resolve().parents[1]
    # Later phases share outputs/ but must not enter this phase's artifact inventory.
    provenance_files = [project_root / name for name in (
        "src/__init__.py", "src/audit_data.py", "src/eda.py", "tests/test_audit_data.py",
        "tests/test_eda.py", "requirements.txt", "notebooks/customer_churn_demo.ipynb",
    )]
    phase_one_artifacts = [output_dir / "data_quality_summary.json"]
    phase_one_artifacts += [tables / name for name in (
        "numeric_distributions.csv", "cohort_memberships.csv", "product_mix.csv",
        "acquisition_mix.csv", "monthly_activity.csv", "customer_type_mix.csv", "target_distribution.csv",
    )]
    phase_one_artifacts += [figures / name for name in (
        "raw_mrr.png", "customer_activity.png", "cohorts.png", "product_mix.png",
        "acquisition_mix.png", "target_distribution.png", "monthly_record_mrr.png",
    )]
    manifest = {
        "phase": 1, "seed": None, "seed_note": "No random operations or model fitting in phase 1.",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {package: importlib.metadata.version(package) for package in (
            "pandas", "numpy", "matplotlib", "scikit-learn", "jupyter", "nbformat", "nbconvert", "ipykernel",
        )},
        "inputs": {path.name: sha256(path) for path in sorted(input_dir.iterdir()) if path.is_file()},
        "code_and_environment": {str(path.relative_to(project_root)): sha256(path) for path in provenance_files if path.is_file()},
        "artifacts": {str(path.relative_to(output_dir)): sha256(path) for path in phase_one_artifacts},
        "reproduce": "python -m src.eda --input-dir input --output-dir outputs",
        "limitations": ["MRR unit is not specified in the assignment.",
                        "Sums of invoice-record MRR fields are exploratory; they are not a verified revenue KPI.",
                        "Product/channel mix is record-weighted, not customer-weighted.",
                        "No supervised feature analysis, cleaning policy or model selection has been performed."],
    }
    write_report(manifest, output_dir / "run_manifest.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    generate_eda(args.input_dir, args.output_dir)
    print(f"Aggregate EDA tables, figures and provenance written to {args.output_dir}")


if __name__ == "__main__":
    main()
