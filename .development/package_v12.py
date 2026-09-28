"""Extend the validated v11 portable archive with the selected v12 answer."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def package():
    manifest = json.loads((ROOT / "artifacts/quality-v12/manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config/solution_v12.json").read_text(encoding="utf-8"))
    assert config["quality_v12"] == manifest
    assert manifest["answer_file"] == "answer.csv"
    assert digest(ROOT / manifest["answer_file"]) == manifest["answer_sha256"]
    assert digest(ROOT / manifest["ranker"]) == manifest["ranker_sha256"]
    assert digest(ROOT / manifest["reference"]) == manifest["reference_sha256"]
    notebook = json.loads((ROOT / "solution.ipynb").read_text(encoding="utf-8"))
    assert notebook["metadata"]["solution_version"] == "v12"

    additions = [
        "solution.ipynb", "answer.csv", "config/solution_v12.json",
        "artifacts/quality-v12/manifest.json",
        "artifacts/quality-v12/ensemble_sweep.json",
        "artifacts/quality-v12/structured_ensemble_sweep.json",
        manifest["ranker"], "docs/EXPERIMENTS_V12.md",
        ".development/finalize_v12.py", ".development/validate_v12.py",
        ".development/quality_v12.py", ".development/package_v12.py",
        ".development/verify_v12.py", ".development/run_validation_notebook.py",
    ]
    replacements = {name: (ROOT / name).read_bytes() for name in additions}
    replacements["README.md"] = (ROOT / "docs/README_V12.md").read_bytes()
    source_path = ROOT / "deliverables/solution_v11.zip"
    target_path = ROOT / "deliverables/solution_v12.zip"
    with zipfile.ZipFile(source_path) as source:
        with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as target:
            for info in source.infolist():
                if info.filename not in replacements and not info.filename.lower().endswith(".ipynb"):
                    with source.open(info) as input_stream, target.open(info.filename, "w", force_zip64=True) as output_stream:
                        shutil.copyfileobj(input_stream, output_stream, 2**20)
            for name, data in replacements.items():
                target.writestr(name, data)
    with zipfile.ZipFile(target_path) as check:
        names = check.namelist()
        assert len(names) == len(set(names))
        assert [name for name in names if name.endswith(".ipynb")] == ["solution.ipynb"]
        assert check.read("answer.csv") == (ROOT / "answer.csv").read_bytes()
        assert check.testzip() is None
    print("Portable v12 archive", target_path, "bytes", target_path.stat().st_size, flush=True)


if __name__ == "__main__":
    package()
