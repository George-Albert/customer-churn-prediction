"""Refit the frozen procedure on all eligible labels and export binary predictions.

June remains evaluation evidence for the earlier fit. This final fit has no new
held-out score. Customer-level outputs and model artifacts remain local/ignored.
"""

import argparse
import importlib.metadata
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.audit_data import CUSTOMER, load_table, sha256, write_report
from src.development_review import CATEGORICAL
from src.prepare_data import KEYS, prepare_frames
from src.train_models import SEED, candidates, preprocessing_state


ALL_LABELED = ("learning", "development", "evaluation")


def validate_submission(predictions, expected_ids):
    """Fail instead of silently losing, duplicating or inventing a submission customer."""
    if list(predictions.columns) != ["FK_DWH_CUSTOMERS", "Predictions"]:
        raise ValueError("Invalid submission headers")
    ids = predictions.FK_DWH_CUSTOMERS
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Invalid or duplicate submission IDs")
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError("Submission ID coverage mismatch")
    if not predictions.Predictions.isin([0, 1]).all():
        raise ValueError("Submission predictions must be binary")


def aggregate_submission(group_scores, expected_ids, threshold):
    """Give each cohort equal weight, then apply one threshold per customer."""
    if group_scores.duplicated(KEYS).any():
        raise ValueError("Duplicate submission cohort scores")
    p = group_scores.probability
    if not np.isfinite(p).all() or not p.between(0, 1).all() or not 0 <= threshold <= 1:
        raise ValueError("Invalid submission probability or threshold")
    customer_scores = group_scores.groupby(CUSTOMER, sort=True).probability.mean()
    predictions = customer_scores.ge(threshold).astype(int).rename("Predictions").reset_index()
    predictions = predictions.rename(columns={CUSTOMER: "FK_DWH_CUSTOMERS"})
    validate_submission(predictions, expected_ids)
    return predictions


def finalize_frames(train, submission, selection):
    """Relearn vocabulary, imputation, encoding and prior using all eligible labeled cohorts."""
    if not 0 <= selection["threshold"] <= 1:
        raise ValueError("Invalid frozen threshold")
    if selection["fallback"]["policy"] != "learning_fallback_prior":
        raise ValueError("Unsupported frozen fallback policy")
    tables, contract = prepare_frames(train, submission, vocabulary_partitions=ALL_LABELED)
    predictors = contract["predictor_columns"]
    # Vocabulary may expand; the approved feature families and formulas must stay fixed.
    base_predictors = [c for c in predictors if not c.startswith("has_product_subcategory_")]
    frozen_base = [c for c in selection["predictor_columns"] if not c.startswith("has_product_subcategory_")]
    if base_predictors != frozen_base:
        raise ValueError("Frozen feature-family contract mismatch")
    numeric = [c for c in predictors if c not in CATEGORICAL]
    model = candidates(numeric).get(selection["selected_candidate"])
    if model is None or model.named_steps["classifier"].get_params(deep=False) != selection["selected_parameters"]:
        raise ValueError("Frozen candidate parameters mismatch")
    features = tables["training_features.csv"].set_index(KEYS)
    metadata = tables["training_metadata.csv"].set_index(KEYS)
    targets = tables["training_targets.csv"].set_index(KEYS).noise
    if not metadata.index.get_level_values(CUSTOMER).is_unique:
        raise ValueError("Eligible labeled customers must have unique identities")
    labels = targets.reindex(features.index)
    if set(labels) != {0, 1}:
        raise ValueError("Final usable labels require both classes")
    fallback_keys = metadata.index[metadata.fallback_required]
    if not len(fallback_keys):
        raise ValueError("Conditional fallback requires labeled fallback customers")
    fallback_labels = targets.reindex(fallback_keys)
    prior = float(fallback_labels.mean())
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features[predictors], labels.to_numpy(dtype=int))
        validation_features = tables["validation_features.csv"].set_index(KEYS)
        group_scores = tables["validation_metadata.csv"].set_index(KEYS).copy()
        group_scores["probability"] = prior
        if len(validation_features):
            positive_column = list(model.classes_).index(1)
            group_scores.loc[validation_features.index, "probability"] = model.predict_proba(
                validation_features[predictors])[:, positive_column]
    group_scores["score_source"] = np.where(group_scores.fallback_required, "conditional_fallback", "model")
    audit = group_scores.reset_index()[KEYS + ["probability", "score_source"]]
    predictions = aggregate_submission(audit, tables["submission_ids.csv"][CUSTOMER], selection["threshold"])
    summary = {"phase": "5", "selected_candidate": selection["selected_candidate"],
               "selected_parameters": selection["selected_parameters"], "threshold": selection["threshold"],
               "seed": SEED, "fitting_customers": len(features), "eligible_labeled_customers": len(metadata),
               "fallback": {"policy": "learning_fallback_prior", "probability": prior,
                            "labeled_customers": len(fallback_keys), "labeled_churn": int(fallback_labels.sum())},
               "vocabulary_partitions": list(ALL_LABELED), "product_vocabulary": contract["product_vocabulary"],
               "predictor_columns": predictors, "encoded_columns": len(model.named_steps["preprocess"].get_feature_names_out()),
               "submission": {**contract["submission"], "predicted_churn": int(predictions.Predictions.sum()),
                              "predicted_retained": int(predictions.Predictions.eq(0).sum()),
                              "multiple_cohort_customers": int(audit.groupby(CUSTOMER).size().gt(1).sum())},
               "identity_exclusions": contract["identity_exclusions"],
               "unseen_product_records": contract["unseen_product_records"],
               "aggregation": "equal mean of cohort probabilities per customer, then probability >= frozen threshold",
               "selection_performed": False, "new_holdout_evaluation": False,
               "limitations": ["June performance belongs to the earlier January-April fit, not this full-data refit.",
                               "Unlabeled 2024 submission outcomes cannot establish prediction accuracy.",
                               "Target construction/availability, annual seasonality and MRR units remain unverified.",
                               "Scores are not claimed to be calibrated individual churn risks."]}
    return {"predictions.csv": predictions, "scoring_audit.csv": audit}, summary, model


def generate_submission(input_dir, model_dir, output_dir, evaluation_dir=Path("outputs/evaluation")):
    """Link approved selection/evaluation and immutable sources to the final local artifacts."""
    root = Path(__file__).resolve().parents[1]
    input_dir, model_dir, output_dir, evaluation_dir = [Path(p).resolve() for p in
                                                       (input_dir, model_dir, output_dir, evaluation_dir)]
    protected = (input_dir, model_dir, evaluation_dir, root / "outputs/preparation")
    if (not output_dir.is_relative_to(root / "outputs") or
            any(output_dir.is_relative_to(p) or p.is_relative_to(output_dir) for p in protected)):
        raise ValueError("Submission output must be under outputs/ and separate from prior stages")
    model_manifest_path, evaluation_manifest_path = model_dir / "model_manifest.json", evaluation_dir / "evaluation_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    evaluation_manifest = json.loads(evaluation_manifest_path.read_text())
    if (set(model_manifest["original_inputs"]) != {"case_study.ipynb", "training.csv", "validation.csv"} or
            not {"selection.json", "selected_pipeline.joblib"}.issubset(model_manifest["artifacts"]) or
            "evaluation_summary.json" not in evaluation_manifest["artifacts"]):
        raise ValueError("Incomplete frozen provenance")
    tracked = {model_manifest_path: sha256(model_manifest_path), evaluation_manifest_path: sha256(evaluation_manifest_path)}
    if evaluation_manifest["model_manifest_sha256"] != tracked[model_manifest_path]:
        raise ValueError("Evaluation does not refer to the frozen selected model")
    for directory, manifest in ((model_dir, model_manifest), (evaluation_dir, evaluation_manifest)):
        for name, digest in manifest["artifacts"].items():
            if Path(name).name != name or sha256(directory / name) != digest:
                raise ValueError("Frozen artifact hash mismatch")
            tracked[directory / name] = digest
    originals = {name: sha256(input_dir / name) for name in model_manifest["original_inputs"]}
    if originals != model_manifest["original_inputs"] or originals != evaluation_manifest["original_inputs"]:
        raise ValueError("Original input hash mismatch")
    versions = {name: importlib.metadata.version(name) for name in model_manifest["packages"]}
    if versions != model_manifest["packages"] or platform.python_version() != model_manifest["python"]:
        raise ValueError("Frozen environment mismatch")
    selection = json.loads((model_dir / "selection.json").read_text())
    evaluation = json.loads((evaluation_dir / "evaluation_summary.json").read_text())
    if (evaluation["selected_candidate"] != selection["selected_candidate"] or
            evaluation["threshold"] != selection["threshold"] or
            evaluation["fallback_probability"] != selection["fallback"]["probability"] or
            evaluation["refit_performed"] or evaluation["selection_performed"]):
        raise ValueError("Frozen evaluation/selection mismatch")
    train, submission = load_table(input_dir / "training.csv", training=True), load_table(input_dir / "validation.csv", training=False)
    tables, summary, model = finalize_frames(train, submission, selection)

    def verify_unchanged():
        if any(sha256(path) != digest for path, digest in tracked.items()) or originals != {
                name: sha256(input_dir / name) for name in originals}:
            raise ValueError("Frozen input changed during final refit")

    verify_unchanged()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, lineterminator="\n", float_format="%.17g")
    reloaded = pd.read_csv(output_dir / "predictions.csv", dtype={"FK_DWH_CUSTOMERS": "string"})
    validate_submission(reloaded, submission[CUSTOMER].unique())
    write_report(summary, output_dir / "submission_summary.json")
    write_report(preprocessing_state(model), output_dir / "final_preprocessing.json")
    joblib.dump(model, output_dir / "final_pipeline.joblib", compress=3)
    verify_unchanged()
    names = [*tables, "submission_summary.json", "final_preprocessing.json", "final_pipeline.joblib"]
    sources = ["src/finalize_submission.py", "src/prepare_data.py", "src/train_models.py", "src/development_review.py",
               "src/audit_data.py", "tests/test_finalize_submission.py", "tests/test_prepare_data.py",
               "notebooks/customer_churn_demo.ipynb", "requirements.txt",
               "README.md", "doc/data_quality.md", "doc/data_dictionary.md", "doc/study_guide.md"]
    write_report({"phase": "5", "python": platform.python_version(), "packages": versions,
                  "seed": SEED, "native_threads": 1, "forest_jobs": 1, "original_inputs": originals,
                  "model_manifest_sha256": tracked[model_manifest_path],
                  "evaluation_manifest_sha256": tracked[evaluation_manifest_path],
                  "frozen_model_artifacts": model_manifest["artifacts"],
                  "frozen_evaluation_artifacts": evaluation_manifest["artifacts"],
                  "code_and_contract": {name: sha256(root / name) for name in sources},
                  "artifacts": {name: sha256(output_dir / name) for name in names},
                  "reproduce": "python -m src.finalize_submission --input-dir input --model-dir outputs/models --output-dir outputs/submission"},
                 output_dir / "submission_manifest.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submission"))
    args = parser.parse_args()
    summary = generate_submission(args.input_dir, args.model_dir, args.output_dir, args.evaluation_dir)
    print(f"Refitted {summary['selected_candidate']} on {summary['fitting_customers']} usable labeled customers; "
          f"exported {summary['submission']['customers']} unique binary predictions at {summary['threshold']:.2f}")


if __name__ == "__main__":
    main()
