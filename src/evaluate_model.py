"""Assess the frozen May-selected procedure on June; never fit or tune.

Only aggregate results are exported. Re-execution reproduces this same assessment,
not a new opportunity to select a model, threshold, fallback or feature policy.
"""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

from src.audit_data import sha256, write_report
from src.development_review import select_review_population
from src.eda import plt
from src.prepare_data import KEYS
from src.train_models import BASELINE, binary_metrics


PREPARED = ("training_features.csv", "training_metadata.csv", "training_targets.csv", "preparation_summary.json")


def evaluate_frames(features, metadata, targets, contract, pipeline, selection, baseline_probability):
    """Use only June values and frozen parameters; baseline comes from verified learning labels."""
    predictors = contract["predictor_columns"]
    if predictors != selection["predictor_columns"] or list(pipeline.feature_names_in_) != predictors:
        raise ValueError("Frozen predictor contract mismatch")
    classifier = pipeline.named_steps["classifier"]
    if classifier.get_params(deep=False) != selection["selected_parameters"]:
        raise ValueError("Frozen classifier parameters mismatch")
    if set(classifier.classes_) != {0, 1}:
        raise ValueError("Frozen classifier requires both classes")
    selected, usable, _ = select_review_population(features, metadata, targets, contract, roles=("evaluation",))
    if selected.empty:
        raise ValueError("Empty June evaluation population")
    # A holdout customer cannot also belong to the fitting/development population.
    customer_counts = metadata.groupby(KEYS[0]).size()
    if customer_counts.reindex(selected.index.get_level_values(KEYS[0])).gt(1).any():
        raise ValueError("Evaluation customer appears in multiple cohorts")
    threshold = selection["threshold"]
    prior = selection["fallback"]["probability"]
    for probability in (prior, baseline_probability):
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid frozen probability")
    probabilities = pd.Series(prior, index=selected.index)
    if len(usable):
        positive_column = list(classifier.classes_).index(1)
        with threadpool_limits(limits=1):
            probabilities.loc[usable.index] = pipeline.predict_proba(usable[predictors])[:, positive_column]
    baseline = pd.Series(baseline_probability, index=selected.index)
    name = selection["selected_candidate"]
    rows, curves = [], []
    for candidate, scores, cutoff in ((name, probabilities, threshold), (BASELINE, baseline, .5)):
        for scope, group in (("all", selected), ("usable", selected.loc[~selected.fallback_required]),
                             ("fallback", selected.loc[selected.fallback_required])):
            rows.append({"candidate": candidate, "scope": scope, "threshold": cutoff,
                         **binary_metrics(group.noise, scores.reindex(group.index), cutoff)})
        if selected.noise.nunique() == 2:
            fpr, tpr, _ = roc_curve(selected.noise, scores)
            curves.extend({"candidate": candidate, "false_positive_rate": float(x), "true_positive_rate": float(y)}
                          for x, y in zip(fpr, tpr))
    metrics = pd.DataFrame(rows)
    summary = {"phase": "4", "population": "June 2023 eligible customers", "selected_candidate": name,
               "threshold": threshold, "fallback_probability": prior, "baseline_probability": baseline_probability,
               "baseline_threshold": .5, "customers": len(selected), "usable_customers": len(usable),
               "fallback_customers": int(selected.fallback_required.sum()), "usable_coverage": len(usable) / len(selected),
               "churn_customers": int(selected.noise.sum()), "retained_customers": int(selected.noise.eq(0).sum()),
               "refit_performed": False, "selection_performed": False,
               "metrics": rows,
               "limitations": ["May supported model, threshold and fallback-policy choices; June cannot retune them.",
                               "Earlier global target EDA was performed; this is not a completely untouched dataset.",
                               "One later cohort does not establish annual seasonality or 2024 submission performance.",
                               "Label construction/availability and MRR accounting units remain unverified.",
                               "Brier score reports probability error; no probability calibration was fitted.",
                               "Final refit and submission scoring have not been performed."]}
    return {"evaluation_metrics.csv": metrics,
            "roc_curve_points.csv": pd.DataFrame(curves, columns=["candidate", "false_positive_rate", "true_positive_rate"])}, summary


def generate_evaluation(preparation_dir, model_dir, output_dir):
    """Verify local provenance before loading the trusted pipeline or parsing June outcomes."""
    root = Path(__file__).resolve().parents[1]
    preparation_dir, model_dir, output_dir = [Path(p).resolve() for p in (preparation_dir, model_dir, output_dir)]
    if (not output_dir.is_relative_to(root / "outputs") or
            any(output_dir.is_relative_to(source) or source.is_relative_to(output_dir)
                for source in (preparation_dir, model_dir))):
        raise ValueError("Evaluation output must be under outputs/ and separate from prepared/model directories")
    manifest_path = model_dir / "model_manifest.json"
    model_manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    required = {"selection.json", "selected_pipeline.joblib", "selected_preprocessing.json", "comparison_metrics.csv"}
    if not required.issubset(manifest["artifacts"]):
        raise ValueError("Frozen model manifest lacks required artifacts")
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or sha256(model_dir / name) != digest:
            raise ValueError("Frozen model artifact hash mismatch")
    preparation_manifest_path = preparation_dir / "preparation_manifest.json"
    preparation_manifest_hash = sha256(preparation_manifest_path)
    if preparation_manifest_hash != manifest["preparation_manifest_sha256"]:
        raise ValueError("Preparation manifest hash mismatch")
    preparation_manifest = json.loads(preparation_manifest_path.read_text())
    hashes = {name: sha256(preparation_dir / name) for name in PREPARED}
    if any(hashes[name] != manifest["prepared_inputs"][name] or
           hashes[name] != preparation_manifest["artifacts"][name] for name in PREPARED):
        raise ValueError("Prepared input hash mismatch")
    originals = {name: sha256(root / "input" / name) for name in manifest["original_inputs"]}
    if originals != manifest["original_inputs"] or originals != preparation_manifest["inputs"]:
        raise ValueError("Original input hash mismatch")
    versions = {name: importlib.metadata.version(name) for name in manifest["packages"]}
    if versions != manifest["packages"] or platform.python_version() != manifest["python"]:
        raise ValueError("Frozen model environment mismatch; use the recorded environment")
    selection = json.loads((model_dir / "selection.json").read_text())
    contract = json.loads((preparation_dir / "preparation_summary.json").read_text())
    # This is a locally generated, hash-verified artifact; never accept external pickle files.
    pipeline = joblib.load(model_dir / "selected_pipeline.joblib")
    frames = [pd.read_csv(preparation_dir / name, dtype="string", keep_default_na=False) for name in PREPARED[:3]]
    features, metadata, targets = frames
    learning_keys = metadata.loc[metadata.partition.eq("learning"), KEYS]
    learning_targets = learning_keys.merge(targets, on=KEYS, how="left", validate="one_to_one")
    labels = pd.to_numeric(learning_targets.noise, errors="raise")
    if labels.empty or not labels.isin([0, 1]).all():
        raise ValueError("Invalid learning labels for frozen baseline")
    # Recover exactly the constant eligible-learning reference; no fit and no June contribution.
    baseline_probability = float(labels.mean())
    tables, summary = evaluate_frames(*frames, contract, pipeline, selection, baseline_probability)
    development = pd.read_csv(model_dir / "comparison_metrics.csv")
    baseline_development = development.loc[development.candidate.eq(BASELINE)].copy()
    selected_development = pd.DataFrame(selection["selected_development_metrics"], index=[0])
    selected_development.insert(0, "threshold", selection["threshold"])
    selected_development.insert(0, "scope", "all")
    selected_development.insert(0, "candidate", selection["selected_candidate"])
    comparison = pd.concat([selected_development, baseline_development.loc[baseline_development.scope.eq("all")]],
                           ignore_index=True)
    comparison.insert(0, "population", "May development")
    june = tables["evaluation_metrics.csv"].query("scope == 'all'").copy()
    june.insert(0, "population", "June evaluation")
    tables["development_comparison.csv"] = pd.concat([comparison, june], ignore_index=True)

    def verify_unchanged():
        if (sha256(manifest_path) != model_manifest_hash or
                sha256(preparation_manifest_path) != preparation_manifest_hash or
                hashes != {name: sha256(preparation_dir / name) for name in PREPARED} or
                originals != {name: sha256(root / "input" / name) for name in originals} or
                any(sha256(model_dir / name) != digest for name, digest in manifest["artifacts"].items())):
            raise ValueError("Frozen input changed during evaluation")

    verify_unchanged()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, na_rep="", lineterminator="\n")
    write_report(summary, output_dir / "evaluation_summary.json")
    fig, ax = plt.subplots(figsize=(7, 5))
    for candidate, curve in tables["roc_curve_points.csv"].groupby("candidate", sort=False):
        auc = tables["evaluation_metrics.csv"].query("scope == 'all'").set_index("candidate").loc[candidate, "roc_auc"]
        ax.plot(curve.false_positive_rate, curve.true_positive_rate, label=f"{candidate} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    if len(tables["roc_curve_points.csv"]):
        ax.legend()
    ax.set(xlabel="False positive rate", ylabel="Churn recall", title="June evaluation ROC; frozen procedure")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_roc.png", dpi=150, metadata={"Software": "customer churn demo frozen evaluation"})
    plt.close(fig)
    result = tables["evaluation_metrics.csv"].iloc[0]
    counts = np.array([[result.tn, result.fp], [result.fn, result.tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(counts, cmap="Blues")
    for (row, column), count in np.ndenumerate(counts):
        ax.text(column, row, str(count), ha="center", va="center", color="black",
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Retained", "Churn"], yticklabels=["Retained", "Churn"],
           xlabel="Predicted", ylabel="Supplied label", title=f"June: probability >= {selection['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_confusion.png", dpi=150, metadata={"Software": "customer churn demo frozen evaluation"})
    plt.close(fig)
    verify_unchanged()
    artifacts = [*tables, "evaluation_summary.json", "evaluation_roc.png", "evaluation_confusion.png"]
    sources = ["src/evaluate_model.py", "src/development_review.py", "src/train_models.py", "src/prepare_data.py",
               "src/audit_data.py", "src/eda.py", "tests/test_evaluate_model.py", "notebooks/customer_churn_demo.ipynb",
               "README.md", "doc/data_quality.md", "doc/study_guide.md", "requirements.txt"]
    write_report({"phase": "4", "python": platform.python_version(), "packages": versions,
                  "refit_performed": False, "selection_performed": False, "native_threads": 1,
                  "model_manifest_sha256": model_manifest_hash,
                  "frozen_model_artifacts": manifest["artifacts"],
                  "fitting_code_and_contract": manifest["code_and_contract"],
                  "prepared_inputs": hashes, "preparation_manifest_sha256": preparation_manifest_hash,
                  "original_inputs": originals, "code_and_contract": {name: sha256(root / name) for name in sources},
                  "artifacts": {name: sha256(output_dir / name) for name in artifacts},
                  "reproduce": "python -m src.evaluate_model --preparation-dir outputs/preparation --model-dir outputs/models --output-dir outputs/evaluation"},
                 output_dir / "evaluation_manifest.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", type=Path, default=Path("outputs/preparation"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    summary = generate_evaluation(args.preparation_dir, args.model_dir, args.output_dir)
    print(f"Evaluated {summary['customers']} June customers at frozen threshold {summary['threshold']:.2f}; no refit or tuning")


if __name__ == "__main__":
    main()
