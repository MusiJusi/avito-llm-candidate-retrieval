"""Check metric semantics, deterministic selection and CSV corruption handling.

Load only the actual function definitions from the notebook; do not run its
training or file-writing cells. No second implementation is used for inference.
"""
import ast
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".inspection_deps"))
import numpy as np
import pandas as pd

notebook = json.loads((root / "Avito.ipynb").read_text(encoding="utf-8"))
namespace = {"np": np, "pd": pd, "re": re}
needed = {"stable_topk", "per_query_recall", "validate_answer"}
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in needed:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook_function", "exec"), namespace)

recall = namespace["per_query_recall"]([[1], [3], [8]], [{1}, {2, 3}, {4}])
assert np.allclose(recall, [1, 0.5, 0]) and recall.mean() == 0.5
assert namespace["per_query_recall"]([[1, 1]], [{1, 2}])[0] == 0.5
topk = namespace["stable_topk"]
assert topk(np.array([1, 3, 3, 3, 0]), 2).tolist() == [1, 2]
assert topk(np.zeros(100), 50).tolist() == list(range(50))

validate = namespace["validate_answer"]
qid = "AbCdEfGhIjKlMn01"
ids = [f"{i:016x}" for i in range(1, 55)]
valid = pd.DataFrame({"query_id": [qid], "answer": [" ".join(ids[:50])]})
assert validate(valid, [qid], ids)
bad_frames = [
    valid.assign(answer=" ".join(ids[:51])),
    valid.assign(answer=ids[0] + " " + ids[0]),
    valid.assign(answer="ABCDEF0123456789"),
    valid.assign(answer="ffffffffffffffff"),
    valid.assign(answer=ids[0] + "  " + ids[1]),
    valid.assign(query_id="abcdEfGhIjKlMn01"),
    valid.assign(extra=1),
    valid.iloc[:0],
    pd.concat([valid, valid], ignore_index=True),
]
for frame in bad_frames:
    try:
        validate(frame, [qid], ids)
    except ValueError:
        continue
    raise AssertionError("Corrupted submission was accepted")
print("Passed: macro recall, deduplication, stable ties, valid CSV, 9 invalid CSV cases")
