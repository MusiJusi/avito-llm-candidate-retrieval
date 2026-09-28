"""Freeze the evaluated v12 model and prepare the standalone notebook."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/bge-m3-v11/final_structured_normal_1d373ec944400cdf.joblib"
REFERENCE = ROOT / "artifacts/bge-m3-v11/benchmark_reference_1d373ec944400cdf.joblib"
ARCHIVED_ANSWER = ROOT / "experiments/results/v12/answer.csv"
ANSWER = ROOT / "answer.csv"
CONFIG = ROOT / "config/solution_v12.json"
NOTEBOOK = ROOT / "solution.ipynb"
MANIFEST = ROOT / "artifacts/quality-v12/manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build():
    assert MODEL.is_file() and REFERENCE.is_file()
    assert ARCHIVED_ANSWER.is_file(), ARCHIVED_ANSWER
    archive_hash = sha256(ARCHIVED_ANSWER)
    assert archive_hash == "f4d83da229b9099fbe2e7725158987bf87bc7c1f68ccafb5c7c3c0e7b2d99fe8"
    if sha256(ANSWER) == archive_hash:
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        assert existing == config["quality_v12"]
        assert existing["answer_file"] == "answer.csv"
        assert notebook["metadata"]["solution_version"] == "v12"
        print("v12 already promoted", archive_hash, flush=True)
        return
    # v11 is retained as an exact, previously submitted fallback.
    old_answer = ROOT / "experiments/results/v11/answer.csv"
    assert sha256(old_answer) == "8ce7401347beb1806f962ae1e72999fa93c89201b1044cd2433494f50e837d7e"
    assert sha256(ANSWER) == sha256(old_answer), "Expected the v11 root answer before promotion"

    config = json.loads((ROOT / "config/solution_v11.json").read_text(encoding="utf-8"))
    selected = {
        "version": "v12",
        "block": "structured",
        "recipe": "normal",
        "weight": 0.5,
        "rrf_constant": 60,
        "trees": 600,
        "ranker": MODEL.relative_to(ROOT).as_posix(),
        "ranker_sha256": sha256(MODEL),
        "reference": REFERENCE.relative_to(ROOT).as_posix(),
        "reference_sha256": sha256(REFERENCE),
        "answer_file": "answer.csv",
        "answer_sha256": archive_hash,
        "selection_report": "artifacts/quality-v12/structured_ensemble_sweep.json",
        "local_matched_recall50": 0.9622251746975228,
        "submitted_v11_local_matched_recall50": 0.9611813862656442,
        "platform_recall50": None,
    }
    config["quality_v12"] = selected
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_hash = sha256(CONFIG)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert "config/solution_v11.json" in "".join(notebook["cells"][42]["source"])
    notebook["cells"][0]["source"] = [
        "# Решение v12: поиск 50 кандидатов по запросу об услугах\n",
        "\n",
        "Кандидат на финальную отправку. Локальный matched Recall@50: **0.962225** "
        "против **0.961181** у отправленного v11. Метрика v12 на платформе пока неизвестна.\n",
        "\n",
        "Notebook воспроизводит корневой `answer.csv` из трёх Parquet-файлов и "
        "сохранённых локальных моделей.\n",
    ]
    config_cell = "".join(notebook["cells"][42]["source"])
    config_cell = config_cell.replace("config/solution_v11.json", "config/solution_v12.json")
    marker = "assert sha256_file(SOLUTION_CONFIG_PATH) == '"
    prefix, rest = config_cell.split(marker, 1)
    _, suffix = rest.split("'", 1)
    notebook["cells"][42]["source"] = (prefix + marker + config_hash + "'" + suffix).splitlines(True)
    notebook["cells"][53]["source"] = ["QUALITY_V11_MANIFEST = SOLUTION_CONFIG['quality_v12']\n"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook.setdefault("metadata", {})["solution_version"] = "v12"
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    # The development answer remains available even after root promotion.
    shutil.copyfile(ARCHIVED_ANSWER, ANSWER)
    MANIFEST.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("v12 promoted", json.dumps({"config_sha256": config_hash,
          "model_sha256": selected["ranker_sha256"], "answer_sha256": archive_hash}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build()
