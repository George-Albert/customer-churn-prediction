"""Publish an allowlisted aggregate snapshot of the default synthetic demo."""

import argparse
import copy
import json
import re
import tempfile
from pathlib import Path

import nbformat

from src.audit_data import sha256
from src.generate_demo_data import generate_demo_data

# Explicit paths prevent accidentally copying row-level tables or serialized models.
FIGURES = {
    "cohorts.png": ("run_manifest.json", "figures/cohorts.png"),
    "mrr_quantiles.png": ("development_review/review_manifest.json", "mrr_quantiles.png"),
    "threshold_search.png": ("models/model_manifest.json", "threshold_search.png"),
    "evaluation_roc.png": ("evaluation/evaluation_manifest.json", "evaluation_roc.png"),
    "evaluation_confusion.png": ("evaluation/evaluation_manifest.json", "evaluation_confusion.png"),
}
MANIFESTS = (
    "run_manifest.json", "preparation/preparation_manifest.json",
    "development_review/review_manifest.json", "models/model_manifest.json",
    "evaluation/evaluation_manifest.json", "submission/submission_manifest.json",
)


def curated_notebook(source, executed):
    """Keep the source cells and aggregate outputs; remove execution timestamps."""
    if [(c.cell_type, c.source) for c in source.cells] != [(c.cell_type, c.source) for c in executed.cells]:
        raise ValueError("Executed notebook does not match the source notebook")
    result = copy.deepcopy(executed)
    images = 0
    for cell in result.cells:
        cell.metadata = {}
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None:
            raise ValueError("Notebook has unexecuted code cells")
        for output in cell.outputs:
            if output.output_type == "error":
                raise ValueError("Notebook contains an execution error")
            data = output.get("data", {})
            if set(data) - {"text/plain", "text/html", "image/png"}:
                raise ValueError("Unexpected notebook output format")
            images += int("image/png" in data)
            text = str(output.get("text", "")) + str(data.get("text/plain", "")) + str(data.get("text/html", ""))
            if re.search(r"SYN-[A-Za-z0-9]|/Users/|/home/|/private/|/tmp/", text):
                raise ValueError("Notebook output contains identifiers or a machine-specific path")
    if images != len(FIGURES):
        raise ValueError("Expected five aggregate notebook figures")
    result.metadata = {k: v for k, v in result.metadata.items() if k in {"kernelspec", "language_info"}}
    return result


def export_demo_results(root=Path(".")):
    """Verify default seed-42 provenance before writing the curated public files."""
    root = Path(root).resolve()
    outputs = root / "outputs"
    # A marker alone is insufficient: require byte equality with freshly generated
    # fictional inputs, never trusting customer data simply because IDs look synthetic.
    with tempfile.TemporaryDirectory() as directory:
        generate_demo_data(directory)
        input_hashes = {p.name: sha256(p) for p in Path(directory).iterdir()}
    for name, digest in input_hashes.items():
        if sha256(root / "input" / name) != digest:
            raise ValueError("Inputs are not the default synthetic demonstration")

    manifests = {}
    for name in MANIFESTS:
        path = outputs / name
        manifest = json.loads(path.read_text())
        recorded = manifest.get("original_inputs", manifest.get("inputs", {}))
        if any(recorded.get(k) != v for k, v in input_hashes.items()):
            raise ValueError(f"Synthetic input provenance mismatch: {name}")
        for key in ("code_and_contract", "code_and_environment"):
            for relative, digest in manifest.get(key, {}).items():
                if sha256(root / relative) != digest:
                    raise ValueError(f"Stale source provenance: {relative}")
        for relative, digest in manifest["artifacts"].items():
            if sha256(path.parent / relative) != digest:
                raise ValueError(f"Artifact hash mismatch: {name}: {relative}")
        manifests[name] = manifest

    source_path = root / "notebooks/customer_churn_demo.ipynb"
    executed_path = outputs / "customer_churn_demo.executed.ipynb"
    notebook = curated_notebook(nbformat.read(source_path, as_version=4),
                                nbformat.read(executed_path, as_version=4))
    evaluation = json.loads((outputs / "evaluation/evaluation_summary.json").read_text())
    final = json.loads((outputs / "submission/submission_summary.json").read_text())
    summary = {
        "data_kind": "synthetic", "seed": 42,
        "interpretation": "Software demonstration only; not observed business performance.",
        "selected_candidate": evaluation["selected_candidate"], "threshold": evaluation["threshold"],
        "assessment_population": evaluation["population"],
        "assessment_metrics": [m for m in evaluation["metrics"] if m["scope"] == "all"],
        "final_fitting_customers": final["fitting_customers"], "scoring_coverage": final["submission"],
        "source_notebook_sha256": sha256(source_path),
        "synthetic_input_sha256": input_hashes,
        "figures_sha256": {},
    }
    # Complete all validation before creating or replacing the publication snapshot.
    figures = {}
    for name, (manifest_name, artifact) in FIGURES.items():
        data = (outputs / manifest_name).parent.joinpath(artifact).read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Not a PNG figure: {name}")
        figures[name] = data
        summary["figures_sha256"][name] = manifests[manifest_name]["artifacts"][artifact]
    destination = root / "doc/demo_results"
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in figures.items():
        (destination / name).write_bytes(data)
    nbformat.write(notebook, destination / "customer_churn_demo.ipynb")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = export_demo_results(args.root)
    print(f"Exported five synthetic figures, aggregate summary and executed notebook: {summary['selected_candidate']}")


if __name__ == "__main__":
    main()
