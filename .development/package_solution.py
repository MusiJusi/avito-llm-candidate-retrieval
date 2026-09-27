"""Package the notebook, pinned model and small reports for offline reproduction.

The supplied Avito Parquet files and installed dependencies are intentionally not
redistributed. The reviewer installs requirements once and supplies the same data.
"""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "artifacts" / "semantic-v1" / "manifest.json").read_text(encoding="utf-8"))
assert hashlib.sha256((root / "answer.csv").read_bytes()).hexdigest() == manifest["answer_sha256"]
destination = root / "deliverables" / "avito_semantic_solution.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
paths = [root / name for name in ["Avito.ipynb", "README.md", "requirements.txt", "CHANGELOG.md", "answer.csv"]]
paths += sorted((root / "models").rglob("*"))
paths += [path for path in (root / "artifacts" / "semantic-v1").iterdir()
          if path.suffix in {".json", ".csv"} or path.name in manifest["final_model_files"]
          or path.name.startswith(("e5_items_", "e5_queries_"))]
paths += [root / ".development" / name for name in ["download_semantic_model.py", "verify_reproduction.py", "run_notebook.py"]]
paths = sorted({path for path in paths if path.is_file() and not path.name.endswith(".tmp")})
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for path in paths:
        print("Packaging", path.relative_to(root), flush=True)
        archive.write(path, path.relative_to(root).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert "models/multilingual-e5-small/model.safetensors" in archive.namelist()
    assert archive.read("answer.csv") == (root / "answer.csv").read_bytes()
print("Ready:", destination, "; MiB:", round(destination.stat().st_size / 2**20, 1), flush=True)
