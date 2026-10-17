"""Fit six bounded churn candidates and select a procedure using May only.

All individual data and the selected pipeline remain in ignored local outputs.
June is excluded by the shared population selector before outcome/amount parsing.
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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.audit_data import sha256, write_report
from src.development_review import CATEGORICAL, select_review_population
from src.eda import plt


SEED = 42
BASELINE = "majority_baseline"
FAMILIES = ("logistic", "tree", "forest")


def candidates(numeric):
    """Fresh pipelines: learning statistics are never reused across experiments."""
    result = {}
    for family in FAMILIES:
        for weight in (None, "balanced"):
            numerical_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
            if family == "logistic":
                numerical_steps.append(("scale", StandardScaler()))
                estimator = LogisticRegression(C=1, solver="lbfgs", max_iter=3000,
                                               class_weight=weight, random_state=SEED)
            elif family == "tree":
                estimator = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50,
                                                   class_weight=weight, random_state=SEED)
            else:
                estimator = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                                   class_weight=weight, random_state=SEED, n_jobs=1)
            preprocessing = ColumnTransformer([
                ("numeric", Pipeline(numerical_steps), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL)),
            ], sparse_threshold=0)
            name = f"{family}_{weight or 'unweighted'}"
            result[name] = Pipeline([("preprocess", preprocessing), ("classifier", estimator)])
    return result


def binary_metrics(y, probabilities, threshold):
    """Churn is positive; undefined one-class AUC/balanced accuracy remain null."""
    y, probabilities = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.shape != probabilities.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid metric label/probability shape or label")
    y = y.astype(int)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid predicted probability")
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    prediction = probabilities >= threshold
    tp, tn = int(((y == 1) & prediction).sum()), int(((y == 0) & ~prediction).sum())
    fp, fn = int(((y == 0) & prediction).sum()), int(((y == 1) & ~prediction).sum())
    both_classes = len(np.unique(y)) == 2
    recall = tp / (tp + fn) if tp + fn else 0.
    retained_recall = tn / (tn + fp) if tn + fp else 0.
    return {"customers": len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tp + tn) / len(y) if len(y) else None,
            "balanced_accuracy": (recall + retained_recall) / 2 if both_classes else None,
            "precision": tp / (tp + fp) if tp + fp else 0., "recall": recall,
            "retained_recall": retained_recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
            "brier_score": float(np.mean((y - probabilities) ** 2)) if len(y) else None}


def choose_candidate(comparison):
    """Use only full-May AUC, then declared exact-tie preferences."""
    ranked = comparison.loc[comparison.scope.eq("all") & comparison.candidate.ne(BASELINE)].copy()
    if ranked.empty or ranked[["roc_auc", "balanced_accuracy"]].isna().any().any():
        raise ValueError("Cannot select without full-development metrics for both classes")
    ranked["family_order"] = ranked.candidate.str.split("_").str[0].map({name: i for i, name in enumerate(FAMILIES)})
    ranked["weight_order"] = ranked.candidate.str.endswith("_balanced").astype(int)
    ranked = ranked.sort_values(["roc_auc", "balanced_accuracy", "family_order", "weight_order"],
                                ascending=[False, False, True, True])
    return ranked.iloc[0].candidate


def choose_threshold(search):
    """Balanced accuracy, then distance to 0.5, then lower threshold."""
    ranked = search.copy()
    # Rounding distance makes symmetric decimal-grid ties independent of float representation.
    ranked["distance"] = (ranked.threshold - .5).abs().round(12)
    if ranked.empty or ranked.balanced_accuracy.isna().any():
        raise ValueError("Cannot select threshold without both development classes")
    return float(ranked.sort_values(["balanced_accuracy", "distance", "threshold"],
                                    ascending=[False, True, True]).iloc[0].threshold)


def train_frames(features, metadata, targets, contract):
    """Fit only learning rows; May supports selection, never preprocessing or priors."""
    selected, usable, numeric = select_review_population(features, metadata, targets, contract)
    predictors = contract["predictor_columns"]
    learning = usable.loc[usable.partition.eq("learning")]
    development = selected.loc[selected.partition.eq("development")]
    if set(learning.noise) != {0, 1} or set(development.noise) != {0, 1}:
        raise ValueError("Usable learning and full development need both classes")
    learning_fallback = selected.loc[selected.partition.eq("learning") & selected.fallback_required]
    if learning_fallback.empty:
        raise ValueError("Conditional fallback requires learning fallback labels; review policy explicitly")
    prior = float(learning_fallback.noise.mean())
    usable_development = usable.loc[usable.partition.eq("development")]
    development_keys = usable_development.index
    fitted = candidates(numeric)
    all_learning = selected.loc[selected.partition.eq("learning")]
    baseline = DummyClassifier(strategy="prior", random_state=SEED)
    baseline.fit(np.zeros((len(all_learning), 1)), all_learning.noise.to_numpy(dtype=int))
    fitted[BASELINE] = baseline
    scores, rows, configs = {}, [], {}
    # Fixed native/forest thread counts and seeds make numerical fitting reproducible.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, model in fitted.items():
            if name == BASELINE:
                probabilities = pd.Series(float(baseline.class_prior_[list(baseline.classes_).index(1)]),
                                          index=development.index)
                configs[name] = baseline.get_params(deep=False)
            else:
                model.fit(learning[predictors], learning.noise.to_numpy(dtype=int))
                classifier = model.named_steps["classifier"]
                probabilities = pd.Series(prior, index=development.index)
                positive_column = list(classifier.classes_).index(1)
                probabilities.loc[development_keys] = model.predict_proba(usable_development[predictors])[:, positive_column]
                configs[name] = classifier.get_params(deep=False)
            scores[name] = probabilities
            for scope, group in (("all", development),
                                 ("usable", development.loc[~development.fallback_required]),
                                 ("fallback", development.loc[development.fallback_required])):
                rows.append({"candidate": name, "scope": scope, "threshold": .5,
                             **binary_metrics(group.noise, probabilities.reindex(group.index), .5)})
    comparison = pd.DataFrame(rows)
    chosen = choose_candidate(comparison)
    search = pd.DataFrame([{"threshold": float(threshold),
                            **binary_metrics(development.noise, scores[chosen], float(threshold))}
                           for threshold in np.linspace(0, 1, 101).round(2)])
    threshold = choose_threshold(search)
    chosen_rows = []
    for stage, cutoff in (("initial_0_5", .5), ("selected", threshold)):
        for scope, group in (("all", development), ("usable", development.loc[~development.fallback_required]),
                             ("fallback", development.loc[development.fallback_required])):
            chosen_rows.append({"candidate": chosen, "stage": stage, "scope": scope, "threshold": cutoff,
                                **binary_metrics(group.noise, scores[chosen].reindex(group.index), cutoff)})
    curves = []
    for name, probabilities in scores.items():
        fpr, tpr, _ = roc_curve(development.noise, probabilities)
        curves.extend({"candidate": name, "false_positive_rate": float(x), "true_positive_rate": float(y)}
                      for x, y in zip(fpr, tpr))
    counts = learning.noise.value_counts()
    classifier = fitted[chosen].named_steps["classifier"]
    selection = {
        "phase": "3B", "seed": SEED, "selected_candidate": chosen, "threshold": threshold,
        "selection_population": "full May development; no June assessment",
        "candidate_rule": "max ROC-AUC; exact ties: balanced accuracy at 0.5, logistic/tree/forest, unweighted/balanced",
        "threshold_rule": "max May balanced accuracy on 0..1 by .01; ties nearest 0.5, then lower; probability >= threshold",
        "predictor_columns": predictors, "product_vocabulary": contract["product_vocabulary"],
        "candidate_parameters": configs, "selected_parameters": classifier.get_params(deep=False),
        "fitting_customers": len(learning), "development_customers": len(development),
        "development_usable_customers": len(usable_development),
        "development_fallback_customers": int(development.fallback_required.sum()),
        "fallback": {"policy": "learning_fallback_prior", "probability": prior,
                     "learning_customers": len(learning_fallback), "learning_churn": int(learning_fallback.noise.sum())},
        "baseline": "constant eligible-learning prevalence; binary learning majority at 0.5; no subgroup override",
        "balanced_weights": {str(int(label)): len(learning) / (2 * int(count)) for label, count in counts.sort_index().items()},
        "selected_development_metrics": binary_metrics(development.noise, scores[chosen], threshold),
        "logistic_iterations": {name: model.named_steps["classifier"].n_iter_.tolist()
                                for name, model in fitted.items() if name.startswith("logistic_")},
        "limitations": ["May supports model, fallback-policy and threshold decisions; its performance is optimistic development evidence.",
                        "No June assessment, full-data refit or submission prediction has been performed.",
                        "Weighted scores are not necessarily calibrated churn probabilities.",
                        "Supplied target construction/availability, MRR units and annual seasonality remain unverified.",
                        "Small metric differences do not establish statistical superiority; no broad search or sensitivity fitted."],
    }
    tables = {"comparison_metrics.csv": comparison, "threshold_search.csv": search,
              "selected_metrics.csv": pd.DataFrame(chosen_rows), "roc_curve_points.csv": pd.DataFrame(curves)}
    return tables, selection, fitted


def preprocessing_state(pipeline):
    """Export learned aggregate statistics/vocabulary to make the fitted representation inspectable."""
    preprocessor = pipeline.named_steps["preprocess"]
    numerical = preprocessor.named_transformers_["numeric"]
    state = {"feature_names": preprocessor.get_feature_names_out().tolist(),
             "numeric_columns": preprocessor.transformers_[0][2],
             "numeric_imputation": numerical.named_steps["impute"].statistics_.tolist(),
             "all_missing_policy": "retain column; learning median imputer uses zero for entirely missing columns",
             "categorical_columns": list(CATEGORICAL),
             "categorical_values": [values.tolist() for values in preprocessor.named_transformers_["categorical"].categories_],
             "unknown_category_policy": "all-zero one-hot block; literal NA is a separate learned category"}
    if "scale" in numerical.named_steps:
        scaler = numerical.named_steps["scale"]
        state["numeric_scaling_mean"], state["numeric_scaling_scale"] = scaler.mean_.tolist(), scaler.scale_.tolist()
    return state


def generate_training(preparation_dir, output_dir):
    """Verify input hashes and persist only aggregate results plus a trusted local pipeline."""
    root = Path(__file__).resolve().parents[1]
    preparation_dir, output_dir = Path(preparation_dir).resolve(), Path(output_dir).resolve()
    if not output_dir.is_relative_to(root / "outputs") or output_dir.is_relative_to(preparation_dir):
        raise ValueError("Model output must be under outputs/ and outside preparation/")
    manifest_path = preparation_dir / "preparation_manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    names = ["training_features.csv", "training_metadata.csv", "training_targets.csv", "preparation_summary.json"]
    hashes = {name: sha256(preparation_dir / name) for name in names}
    if any(hashes[name] != manifest["artifacts"][name] for name in names):
        raise ValueError("Prepared input hash mismatch; regenerate preparation")
    originals = {name: sha256(root / "input" / name) for name in manifest["inputs"]}
    if originals != manifest["inputs"]:
        raise ValueError("Original input hash differs from preparation provenance")
    contract = json.loads((preparation_dir / "preparation_summary.json").read_text())
    frames = [pd.read_csv(preparation_dir / name, dtype="string", keep_default_na=False) for name in names[:3]]
    tables, selection, fitted = train_frames(*frames, contract)
    if hashes != {name: sha256(preparation_dir / name) for name in names} or sha256(manifest_path) != manifest_hash:
        raise ValueError("Prepared input changed during fitting")
    if originals != {name: sha256(root / "input" / name) for name in originals}:
        raise ValueError("Original input changed during fitting")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, na_rep="", lineterminator="\n")
    pipeline = fitted[selection["selected_candidate"]]
    write_report(selection, output_dir / "selection.json")
    write_report(preprocessing_state(pipeline), output_dir / "selected_preprocessing.json")
    # This local artifact is hash-inventoried. Never load untrusted external pickle/joblib files.
    joblib.dump(pipeline, output_dir / "selected_pipeline.joblib", compress=3)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, group in tables["roc_curve_points.csv"].groupby("candidate", sort=False):
        auc = tables["comparison_metrics.csv"].query("scope == 'all'").set_index("candidate").loc[name, "roc_auc"]
        ax.plot(group.false_positive_rate, group.true_positive_rate, label=f"{name} ({auc:.3f})")
    ax.set(xlabel="False positive rate", ylabel="Churn recall", title="May development ROC; full eligible population")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "development_roc.png", dpi=150, metadata={"Software": "customer churn demo bounded model comparison"})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    search = tables["threshold_search.csv"]
    ax.plot(search.threshold, search.balanced_accuracy)
    ax.axvline(selection["threshold"], color="#d17732", label=f"Selected: {selection['threshold']:.2f}")
    ax.axvline(.5, linestyle="--", color="#286487", label="Initial reference: 0.50")
    ax.set(xlabel="Probability threshold", ylabel="Balanced accuracy",
           title=f"May threshold selection: {selection['selected_candidate']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_search.png", dpi=150, metadata={"Software": "customer churn demo bounded model comparison"})
    plt.close(fig)
    artifacts = [*tables, "selection.json", "selected_preprocessing.json", "selected_pipeline.joblib",
                 "development_roc.png", "threshold_search.png"]
    sources = ["src/train_models.py", "src/development_review.py", "src/prepare_data.py", "src/audit_data.py", "src/eda.py",
               "tests/test_train_models.py", "tests/test_development_review.py", "notebooks/customer_churn_demo.ipynb",
               "README.md", "doc/data_quality.md", "doc/data_dictionary.md", "doc/study_guide.md", "requirements.txt"]
    write_report({"phase": "3B", "seed": SEED, "native_threads": 1, "forest_jobs": 1,
                  "python": platform.python_version(),
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("pandas", "numpy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl")},
                  "original_inputs": originals, "prepared_inputs": hashes, "preparation_manifest_sha256": manifest_hash,
                  "code_and_contract": {name: sha256(root / name) for name in sources},
                  "artifacts": {name: sha256(output_dir / name) for name in artifacts},
                  "reproduce": "python -m src.train_models --preparation-dir outputs/preparation --output-dir outputs/models"},
                 output_dir / "model_manifest.json")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", type=Path, default=Path("outputs/preparation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models"))
    args = parser.parse_args()
    selection = generate_training(args.preparation_dir, args.output_dir)
    print(f"Selected {selection['selected_candidate']} at {selection['threshold']:.2f} on May; no June assessment")


if __name__ == "__main__":
    main()
