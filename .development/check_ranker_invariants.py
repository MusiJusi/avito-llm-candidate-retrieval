"""Leakage barriers, negative sampling and unchanged retrieval are contracts.

Load real notebook definitions without executing training cells. These checks
exercise adversarial overlaps rather than duplicating the retrieval algorithm.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".inspection_deps"))
import numpy as np
import pandas as pd
from scipy.stats import rankdata

notebook = json.loads((root / "Avito.ipynb").read_text(encoding="utf-8"))
functions = {}
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                functions[node.name] = node

base_source = subprocess.check_output(
    ["git", "show", "v0.1.0:.development/notebook_source.py"], cwd=root,
).decode("utf-8")
baseline = {node.name: node for node in ast.parse(base_source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
for name in ["BM25Index", "HistorySignals", "retrieve_features", "score_candidates", "normalize_text"]:
    assert ast.dump(functions[name]) == ast.dump(baseline[name]), f"Baseline retrieval changed: {name}"
assert "search_category" not in ast.unparse(functions["rank_features"])
assert "item_category_id" not in ast.unparse(functions["rank_features"])

namespace = {"np": np, "pd": pd, "rankdata": rankdata}
for name in ["stable_topk", "score_candidates", "predict_from_features", "purge_history", "hard_negative_sample", "blend_predictions"]:
    exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), "notebook_function", "exec"), namespace)

namespace["CONFIG"] = {"cold_item_fraction": 1.0}
namespace["ITEM_IDS"] = np.array(["A", "B", "C", "D"])
history = pd.DataFrame({"query_norm": ["held", "held", "other", "safe", "safe2"],
                        "item_id": ["A", "B", "A", "C", "D"]})
queries = pd.DataFrame({"query_norm": ["held"]})
clean, purged = namespace["purge_history"](history, queries, [{0, 1}], 42)
assert set(clean.query_norm) == {"safe", "safe2"}
assert purged == {"A", "B"}
# A target item from a DIFFERENT query must also disappear when item-purged.
assert "other" not in set(clean.query_norm)

namespace["RANKER_CONFIG"] = {"hard_negatives": 2, "random_negatives": 1}
namespace["best_config"] = {"title": .3, "body": .5, "char": .2, "filter": .03, "geo": .5, "micro": .5}
ids = np.arange(10)
features = np.ones((10, 6), dtype=np.float32)
truth = {1, 7, 99}  # 99 was not retrieved: it must never be injected.
selected, target = namespace["hard_negative_sample"](ids, features, truth, np.random.default_rng(42))
assert {1, 7}.issubset(set(ids[selected]))
assert len(selected) == 5 and 99 not in ids[selected]
assert not set(ids[selected[~target[selected]]]) & truth
selected2, _ = namespace["hard_negative_sample"](ids, features, truth, np.random.default_rng(42))
assert np.array_equal(selected, selected2)
missing, _ = namespace["hard_negative_sample"](ids, features, {99}, np.random.default_rng(42))
assert len(missing) == 0

records = [(ids, features)]
prediction = namespace["blend_predictions"](records, [np.arange(10)], 0, 3)
expected = namespace["predict_from_features"](records, namespace["best_config"], 3)
assert np.array_equal(prediction[0], expected[0])
print("Passed: unchanged retrievers, no category scoring, text/item purge, no label injection, multi-positive sampling, deterministic sampling, baseline fallback")
