"""Reproduce the packaged solution in a fresh directory with CUDA hidden.

Only declared deliverables are copied: E5 vectors, weights and final fitted model.
No lexical indexes, feature matrices or evaluation models are reused. Supplied
Parquet files are linked read-only in practice: the notebook only reads them.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

root = Path(__file__).resolve().parents[1]
archive_path = root / "deliverables" / "avito_semantic_solution.zip"
reference = json.loads((root / "artifacts" / "semantic-v1" / "manifest.json").read_text(encoding="utf-8"))
verification_root = root / ".verification"
verification_root.mkdir(exist_ok=True)
destination = Path(tempfile.mkdtemp(prefix="portable_cpu_", dir=verification_root))
with zipfile.ZipFile(archive_path) as archive:
    for name in archive.namelist():
        resolved = (destination / name).resolve()
        assert resolved.is_relative_to(destination.resolve()), "Archive path escapes verification workspace"
    archive.extractall(destination)
for name in ["train.parquet", "benchmark_items.parquet", "benchmark_queries.parquet"]:
    os.link(root / name, destination / name)
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "-1"
env["AVITO_DISABLE_NETWORK"] = "1"
env["AVITO_REBUILD_CACHE"] = "0"
env["PYTHONIOENCODING"] = "utf-8"
# Equivalent to installing requirements in the reviewer's kernel environment.
# The submission itself does not refer to these local dependency directories.
env["PYTHONPATH"] = os.pathsep.join([str(root / ".semantic_deps"), str(root / ".inspection_deps")])
started = time.perf_counter()
subprocess.run([sys.executable, "-u", str(destination / ".development" / "run_notebook.py")],
               cwd=destination, env=env, check=True)
result = json.loads((destination / "artifacts" / "semantic-v1" / "manifest.json").read_text(encoding="utf-8"))
digest = hashlib.sha256((destination / "answer.csv").read_bytes()).hexdigest()
assert result["versions"]["device"] == "cpu"
assert digest == reference["answer_sha256"], "Packaged CPU reproduction changed the submission"
assert result["metrics"] == reference["metrics"]
assert result["code_sha256"] == reference["code_sha256"]
report = {"fresh_extracted_directory": True, "device": "cpu", "cuda_visible_devices": "-1",
          "python_network_connections_disabled": True, "original_search_caches_available": False,
          "declared_frozen_e5_vectors_used": True, "declared_final_model_used": True,
          "answer_bytes_equal": True, "metrics_equal": True, "answer_sha256": digest,
          "elapsed_seconds": round(time.perf_counter() - started, 2)}
(root / "artifacts" / "semantic-v1" / "portable_reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
print("Verification directory:", destination, flush=True)
