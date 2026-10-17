"""Review learning/development aggregates without fitting a churn classifier.

June is excluded before reading outcomes or numerical feature values. Original
and customer-level files stay untouched; only aggregate artifacts are written.
"""

import argparse
import importlib.metadata
import json
import platform
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_data import sha256, write_report
from src.eda import plt
from src.prepare_data import KEYS, PARTITIONS


ROLES = ("learning", "development")
CATEGORICAL = ("origin_group", "origin_subgroup")


def select_review_population(features, metadata, targets, contract, *, roles=ROLES):
    """Validate selected prepared partitions; defaults preserve January-May review."""
    predictors = contract["predictor_columns"]
    forbidden = set(KEYS + ["noise", "partition", "fallback_required", "unseen_product_records"])
    if forbidden.intersection(predictors) or len(set(predictors)) != len(predictors):
        raise ValueError("Invalid predictor contract")
    if list(features.columns) != KEYS + predictors:
        raise ValueError("Feature schema differs from preparation contract")
    for frame in (features, metadata, targets):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate customer/cohort keys")

    selected = metadata.loc[metadata.partition.isin(roles)].set_index(KEYS).copy()
    if not selected.index.get_level_values(KEYS[0]).is_unique:
        raise ValueError("Customer appears in multiple review cohorts")
    expected_roles = selected.index.get_level_values("customer_start_date").map(PARTITIONS)
    if not (selected.partition.to_numpy() == expected_roles.to_numpy()).all():
        raise ValueError("Cohort/partition mismatch")
    if not selected.fallback_required.isin(["True", "False"]).all():
        raise ValueError("Invalid fallback flag")
    selected["fallback_required"] = selected.fallback_required.eq("True")
    target_table = targets.set_index(KEYS)
    if not selected.index.isin(target_table.index).all():
        raise ValueError("Missing selected targets")
    # Parse only selected outcomes/amounts; default callers still exclude June.
    selected["noise"] = pd.to_numeric(target_table.reindex(selected.index).noise, errors="raise")
    if not selected.noise.isin([0, 1]).all():
        raise ValueError("Selected targets must be binary")
    usable_keys = selected.index[~selected.fallback_required]
    feature_table = features.set_index(KEYS)
    present_keys = feature_table.index[feature_table.index.isin(selected.index)]
    if set(present_keys) != set(usable_keys):
        raise ValueError("Feature/fallback coverage mismatch")
    usable = selected.loc[usable_keys].join(feature_table.reindex(usable_keys))
    numeric = [column for column in predictors if column not in CATEGORICAL]
    for column in numeric:
        usable[column] = pd.to_numeric(usable[column].replace("", pd.NA), errors="raise").astype(float)
        if np.isinf(usable[column]).any():
            raise ValueError("Nonfinite feature value")
    return selected, usable, numeric


def review_frames(features, metadata, targets, contract):
    """Validate selected populations and return aggregates, never individual scores."""
    predictors = contract["predictor_columns"]
    selected, usable, numeric = select_review_population(features, metadata, targets, contract)

    classes, cohorts, distributions, categories, months = [], [], [], [], []
    for role in ROLES:
        population = selected.loc[selected.partition.eq(role)]
        rows = usable.loc[usable.partition.eq(role)]
        if population.empty or rows.empty:
            raise ValueError("Each review partition needs eligible and usable customers")
        for scope, group in (("all", population), ("usable", rows),
                             ("fallback", population.loc[population.fallback_required])):
            n, churn = len(group), int(group.noise.sum())
            classes.append({"partition": role, "scope": scope, "customers": n,
                            "retained": n - churn, "churn": churn,
                            "churn_rate": churn / n if n else None})
        for cohort, group in population.groupby(level="customer_start_date", sort=True):
            cohorts.append({"partition": role, "cohort": cohort, "customers": len(group),
                            "churn": int(group.noise.sum()), "churn_rate": float(group.noise.mean())})
        for column in numeric:
            values = rows[column]
            distributions.append({"partition": role, "feature": column, "customers": len(rows),
                                  "missing": int(values.isna().sum()), "mean": values.mean(),
                                  "min": values.min(), "p01": values.quantile(.01),
                                  "p25": values.quantile(.25), "median": values.median(),
                                  "p75": values.quantile(.75), "p99": values.quantile(.99),
                                  "max": values.max()})
        for column in CATEGORICAL:
            for category, count in rows[column].value_counts().sort_index().items():
                categories.append({"partition": role, "feature": column, "category": category,
                                   "customers": int(count), "share": count / len(rows)})
        for (month_0, month_1), group in rows.groupby(["month_0_observed", "month_1_observed"], sort=True):
            months.append({"partition": role, "month_0_observed": int(month_0),
                           "month_1_observed": int(month_1), "customers": len(group),
                           "churn": int(group.noise.sum()), "churn_rate": float(group.noise.mean())})

    learning = usable.loc[usable.partition.eq("learning")]
    duplicates = [{"left": left, "right": right} for left, right in combinations(numeric, 2)
                  if learning[left].equals(learning[right])]
    conditional = selected.loc[selected.partition.eq("learning") & selected.fallback_required]
    development_fallback = selected.loc[selected.partition.eq("development") & selected.fallback_required]
    priors = {"global_learning_prior": float(selected.loc[selected.partition.eq("learning"), "noise"].mean())}
    if not conditional.empty:
        priors["learning_fallback_prior"] = float(conditional.noise.mean())
    comparisons = []
    for policy, probability in priors.items():
        y = development_fallback.noise
        prediction = int(probability >= .5)
        tp, fn = int(y.sum()) * prediction, int(y.sum()) * (1 - prediction)
        tn, fp = int(y.eq(0).sum()) * (1 - prediction), int(y.eq(0).sum()) * prediction
        comparisons.append({"policy": policy, "learning_customers": len(conditional) if policy == "learning_fallback_prior"
                            else int(selected.partition.eq("learning").sum()),
                            "probability": probability, "fixed_threshold": .5,
                            "development_fallback_customers": len(y), "prediction": prediction,
                            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                            "accuracy": (tn + tp) / len(y) if len(y) else np.nan,
                            "brier_score": float(((y - probability) ** 2).mean()) if len(y) else np.nan})
    shifts = []
    for column, category in contract["product_vocabulary"].items():
        left = float(learning[column].mean())
        right = float(usable.loc[usable.partition.eq("development"), column].mean())
        shifts.append({"feature": column, "product_subcategory": category, "learning_share": left,
                       "development_share": right, "difference_percentage_points": 100 * (right - left)})
    usable_counts = learning.noise.value_counts()
    if set(usable_counts.index) != {0, 1}:
        raise ValueError("Usable learning population needs both classes")
    summary = {
        "phase": "3A", "partitions_reviewed": list(ROLES), "model_fitted": False,
        "class_distribution": classes,
        "balanced_weights_usable_learning": {str(int(label)): len(learning) / (2 * int(count))
                                            for label, count in usable_counts.sort_index().items()},
        "constant_learning_predictors": [name for name in predictors if learning[name].nunique(dropna=False) == 1],
        "origin_subgroup_determines_group_in_learning": bool(
            learning.groupby("origin_subgroup").origin_group.nunique(dropna=False).le(1).all()),
        "unseen_development_acquisition_categories": {
            name: sorted(set(usable.loc[usable.partition.eq("development"), name]) - set(learning[name]))
            for name in CATEGORICAL},
        "fallback_primary": "Phase-3A checkpoint: global_learning_prior; phase-3B adoption is recorded in selection.json",
        "limitations": ["June outcome/feature statistics excluded; earlier global phase-1 target EDA is documented.",
                        "Fallback comparison uses fixed 0.5; no threshold or primary policy selected.",
                        "Small subgroups and cohort shifts do not establish causes or annual seasonality.",
                        "No imputation, scaling, encoding, class weighting or churn estimator fitted."],
    }
    tables = {"class_distribution.csv": pd.DataFrame(classes), "cohort_distribution.csv": pd.DataFrame(cohorts),
              "numeric_distributions.csv": pd.DataFrame(distributions),
              "categorical_distributions.csv": pd.DataFrame(categories), "month_coverage.csv": pd.DataFrame(months),
              "learning_duplicate_features.csv": pd.DataFrame(duplicates, columns=["left", "right"]),
              "fallback_comparison.csv": pd.DataFrame(comparisons), "product_presence_shift.csv": pd.DataFrame(shifts)}
    return tables, summary


def generate_review(preparation_dir, output_dir):
    """Verify prepared inputs, then save deterministic aggregates and provenance."""
    preparation_dir, output_dir = Path(preparation_dir).resolve(), Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[1]
    if not output_dir.is_relative_to(root / "outputs") or output_dir.is_relative_to(preparation_dir):
        raise ValueError("Review output must be under outputs/ and outside preparation/")
    names = ["training_features.csv", "training_metadata.csv", "training_targets.csv", "preparation_summary.json"]
    manifest_path = preparation_dir / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {name: sha256(preparation_dir / name) for name in names}
    if any(hashes[name] != manifest["artifacts"][name] for name in names):
        raise ValueError("Prepared input hash mismatch; regenerate preparation")
    contract = json.loads((preparation_dir / "preparation_summary.json").read_text())
    frames = [pd.read_csv(preparation_dir / name, dtype="string", keep_default_na=False) for name in names[:3]]
    tables, summary = review_frames(*frames, contract)
    if hashes != {name: sha256(preparation_dir / name) for name in names}:
        raise ValueError("Prepared input changed during review")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, na_rep="", lineterminator="\n")
    write_report(summary, output_dir / "review_summary.json")
    counts = tables["class_distribution.csv"].query("scope == 'all'").set_index("partition")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.retained, label="Retained (0)", color="#286487")
    ax.bar(counts.index, counts.churn, bottom=counts.retained, label="Churn (1)", color="#d17732")
    ax.set(ylabel="Eligible customers", title="Learning/development: customer-level classes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150, metadata={"Software": "customer churn demo development review"})
    plt.close(fig)
    stats = tables["numeric_distributions.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for month, ax in enumerate(axes):
        boxes = []
        for role in ROLES:
            row = stats.loc[stats.partition.eq(role) & stats.feature.eq(f"invoice_record_mrr_sum_month_{month}")].iloc[0]
            boxes.append({"label": role, "med": row["median"], "q1": row.p25, "q3": row.p75,
                          "whislo": row.p01, "whishi": row.p99, "fliers": []})
        ax.bxp(boxes, showfliers=False)
        ax.set_yscale("symlog", linthresh=.01)
        ax.set(title=f"Month {month}: usable customers", ylabel="Signed record-MRR sum (source units)")
    fig.suptitle("Quartiles; whiskers = 1st/99th percentiles; tails omitted for display")
    fig.tight_layout()
    fig.savefig(output_dir / "mrr_quantiles.png", dpi=150, metadata={"Software": "customer churn demo development review"})
    plt.close(fig)
    artifacts = [*tables, "review_summary.json", "class_distribution.png", "mrr_quantiles.png"]
    sources = ["src/development_review.py", "src/eda.py", "src/prepare_data.py", "src/audit_data.py",
               "tests/test_development_review.py", "tests/test_audit_data.py", "notebooks/customer_churn_demo.ipynb",
               "doc/study_guide.md", "doc/data_quality.md", "doc/data_dictionary.md", "requirements.txt"]
    write_report({"phase": "3A", "seed": None, "python": platform.python_version(),
                  "packages": {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "matplotlib")},
                  "prepared_inputs": hashes, "preparation_manifest_sha256": sha256(manifest_path),
                  "original_inputs": manifest["inputs"],
                  "code_and_contract": {name: sha256(root / name) for name in sources},
                  "artifacts": {name: sha256(output_dir / name) for name in artifacts},
                  "reproduce": "python -m src.development_review --preparation-dir outputs/preparation --output-dir outputs/development_review"},
                 output_dir / "review_manifest.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", type=Path, default=Path("outputs/preparation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/development_review"))
    args = parser.parse_args()
    generate_review(args.preparation_dir, args.output_dir)
    print(f"Phase-3A learning/development aggregates written to {args.output_dir}; no churn model fitted")


if __name__ == "__main__":
    main()
