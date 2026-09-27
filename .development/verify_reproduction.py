"""Rebuild the complete notebook in a fresh process and compare the submission.

Run after an initial successful execution. Search indexes, OOF examples and all
models are rebuilt: the check does not merely load the previously fitted model.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import nbformat

root = Path(__file__).resolve().parents[1]
artifacts = root / "artifacts" / "learned-selection-v1"
before = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
expected_hash = hashlib.sha256((root / "answer.csv").read_bytes()).hexdigest()
assert before["answer_sha256"] == expected_hash
env = os.environ.copy()
env["AVITO_REBUILD_CACHE"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
started = time.perf_counter()
subprocess.run([sys.executable, "-u", str(root / ".development" / "run_notebook.py")],
               cwd=root, env=env, check=True)
after = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
actual_hash = hashlib.sha256((root / "answer.csv").read_bytes()).hexdigest()
assert expected_hash == actual_hash, "Full rebuild changed the submitted CSV"
assert before["code_sha256"] == after["code_sha256"]
assert before["metrics"] == after["metrics"], "Full rebuild changed experiment results"
notebook = nbformat.read(root / "Avito.ipynb", as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
report = {"full_rebuild_without_cache": True, "fresh_python_process": True,
          "all_models_retrained": True, "answer_bytes_equal": True, "metrics_equal": True,
          "answer_sha256": actual_hash, "code_sha256": after["code_sha256"],
          "executed_code_cells": len(code_cells), "elapsed_seconds": round(time.perf_counter() - started, 2)}
(artifacts / "reproducibility_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
