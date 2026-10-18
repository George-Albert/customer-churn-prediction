"""Safety and determinism checks for the public aggregate snapshot exporter."""

import json
import tempfile
import unittest
from pathlib import Path

import nbformat

from src.audit_data import sha256
from src.export_demo_results import FIGURES, MANIFESTS, export_demo_results
from src.generate_demo_data import generate_demo_data


class ExportDemoResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        generate_demo_data(self.root / "input")
        (self.root / "notebooks").mkdir()
        source = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('aggregate')")])
        self.source = self.root / "notebooks/customer_churn_demo.ipynb"
        nbformat.write(source, self.source)
        source.cells[0].execution_count = 1
        source.cells[0].metadata["execution"] = {"timestamp": "variable"}
        source.cells[0].outputs = [nbformat.v4.new_output("display_data", data={"image/png": "cG5n"}) for _ in FIGURES]
        self.executed = self.root / "outputs/customer_churn_demo.executed.ipynb"
        self.executed.parent.mkdir()
        nbformat.write(source, self.executed)
        input_hashes = {p.name: sha256(p) for p in (self.root / "input").iterdir()}
        for name in MANIFESTS:
            path = self.root / "outputs" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"original_inputs": input_hashes,
                                       "code_and_contract": {"notebooks/customer_churn_demo.ipynb": sha256(self.source)},
                                       "artifacts": {}}))
        for _, (manifest, artifact) in FIGURES.items():
            path = self.root / "outputs" / manifest
            figure = path.parent / artifact
            figure.parent.mkdir(parents=True, exist_ok=True)
            figure.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            self.record(path, artifact)
        for relative, data in {
            "evaluation/evaluation_summary.json": {"selected_candidate": "fixture", "threshold": 0.4,
                "population": "fictional", "metrics": [{"scope": "all", "roc_auc": 0.5}]},
            "submission/submission_summary.json": {"fitting_customers": 10, "submission": {"customers": 5}},
        }.items():
            path = self.root / "outputs" / relative
            path.write_text(json.dumps(data))
            manifest = path.parent / ("evaluation_manifest.json" if "evaluation/" in relative else "submission_manifest.json")
            self.record(manifest, path.name)

    def record(self, manifest_path, artifact):
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][artifact] = sha256(manifest_path.parent / artifact)
        manifest_path.write_text(json.dumps(manifest))

    def test_allowlist_repeatability_and_metadata_removal(self):
        export_demo_results(self.root)
        destination = self.root / "doc/demo_results"
        first = {p.name: sha256(p) for p in destination.iterdir()}
        self.assertEqual(set(first), {*FIGURES, "summary.json", "customer_churn_demo.ipynb"})
        export_demo_results(self.root)
        self.assertEqual(first, {p.name: sha256(p) for p in destination.iterdir()})
        snapshot = nbformat.read(destination / "customer_churn_demo.ipynb", as_version=4)
        self.assertEqual(snapshot.cells[0].metadata, {})
        self.assertTrue(json.loads((destination / "summary.json").read_text())["data_kind"] == "synthetic")

    def test_modified_synthetic_inputs_are_rejected(self):
        path = self.root / "input/training.csv"
        path.write_text(path.read_text() + "modified")
        with self.assertRaisesRegex(ValueError, "default synthetic"):
            export_demo_results(self.root)
        self.assertFalse((self.root / "doc").exists())

    def test_stale_source_is_rejected(self):
        self.source.write_text(self.source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Stale source"):
            export_demo_results(self.root)

    def test_tampered_figure_is_rejected(self):
        (self.root / "outputs/evaluation/evaluation_roc.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Artifact hash"):
            export_demo_results(self.root)

    def test_incomplete_errors_ids_and_changed_code_are_rejected(self):
        original = nbformat.read(self.executed, as_version=4)
        for condition in ("incomplete", "error", "identifier", "path", "source", "missing_figure"):
            with self.subTest(condition=condition):
                notebook = nbformat.from_dict(json.loads(nbformat.writes(original)))
                cell = notebook.cells[0]
                if condition == "incomplete":
                    cell.execution_count = None
                elif condition == "error":
                    cell.outputs.append(nbformat.v4.new_output("error", ename="Error", evalue="failure", traceback=[]))
                elif condition in ("identifier", "path"):
                    text = "SYN-0001" if condition == "identifier" else "/Users/private/data"
                    cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
                elif condition == "source":
                    cell.source = "changed()"
                else:
                    cell.outputs.pop()
                nbformat.write(notebook, self.executed)
                with self.assertRaises(ValueError):
                    export_demo_results(self.root)
                self.assertFalse((self.root / "doc").exists())


if __name__ == "__main__":
    unittest.main()
