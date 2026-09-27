"""Check v0.3 exact cosine, pool expansion, ID alignment and negative sampling."""
import ast
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".inspection_deps"))
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

recipe = root / "Avito_v0.3.ipynb"
document = json.loads((recipe if recipe.exists() else root / "Avito.ipynb").read_text(encoding="utf-8"))
definitions = {}
for cell in document["cells"]:
    if cell["cell_type"] == "code":
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = node
namespace = {"np": np, "time": time, "rankdata": rankdata}
needed = ["stable_topk", "normalized_scores", "score_candidates", "score_lexical_columns", "semantic_affinity",
          "retrieve_semantic_features", "semantic_negative_sample", "rank_semantic_features",
          "predict_from_features", "blend_predictions", "ensemble_predictions"]
for name in needed:
    exec(compile(ast.Module(body=[definitions[name]], type_ignores=[]), "notebook", "exec"), namespace)
namespace["legacy_score_candidates"] = namespace["score_candidates"]
namespace["score_candidates"] = namespace["score_lexical_columns"]
namespace["SEMANTIC_CONFIG"] = {"temperature": .05, "retrieval_pool": 1}
namespace["CONFIG"] = {"retrieval_pool": 1}
namespace["RANKER_CONFIG"] = {"hard_negatives": 2, "random_negatives": 1}
namespace["best_config"] = {"title": .3, "body": .5, "char": .2, "filter": .03, "geo": .5, "micro": .5}
scores = np.array([1, .8, .2, .1, 0, 0], dtype=np.float32)
class Queries:
    def query_matrix(self, texts):
        return sp.csr_matrix(np.ones((len(texts), 1), dtype=np.float32))
    def transform(self, texts):
        return self.query_matrix(texts)
class History:
    def micro_predictions(self, texts):
        return np.ones((len(texts), 1), dtype=np.float32)
    def geography(self, loc):
        return np.array([1, 1, 1, 1, .01, 1], dtype=np.float32)
class Semantic:
    def scores(self, texts):
        return np.tile([.3, .3, .3, .3, .95, .9], (len(texts), 1)).astype(np.float32)
namespace["lexical"] = {"title": SimpleNamespace(query_matrix=Queries().query_matrix, transpose=sp.csr_matrix(scores[None, :])),
                        "body": SimpleNamespace(query_matrix=Queries().query_matrix, transpose=sp.csr_matrix(scores[None, :])),
                        "char_vectorizer": Queries(), "char_transpose": sp.csr_matrix(scores[None, :])}
namespace["ITEM_MICRO_COLS"] = np.zeros(6, dtype=int)
namespace["semantic_index"] = Semantic()
queries = pd.DataFrame({"query_norm": ["q"], "search_infm_params_text": [""], "search_location_id": [1]})
records = namespace["retrieve_semantic_features"](queries, History())
ids, features = records[0]
assert set(ids) == {0, 4, 5}, "Old lexical candidate or new semantic candidate was dropped"
assert ids.tolist() == [0, 4, 5], "Item row ordering changed"
assert features.shape == (3, 10)
assert features[:, 8].tolist() == [1, 0, 0]
assert features[:, 9].tolist() == [0, 1, 1]
assert np.allclose(features[:, 6], [.3, .95, .9])
assert np.isfinite(features).all()
for truth in [{4}, {0, 5}, {99}]:
    selected, target = namespace["semantic_negative_sample"](ids, features, truth, np.random.default_rng(7))
    if truth == {99}:
        assert len(selected) == 0
    else:
        assert truth.issubset(set(ids[selected]))
        assert not set(ids[selected[~target[selected]]]) & truth
        assert 99 not in ids[selected]

namespace["legacy_rank_features"] = lambda query, record: np.zeros((len(record[0]), 32), dtype=np.float32)
namespace["RANK_FEATURE_NAMES"] = list(range(38))
matrix = namespace["rank_semantic_features"](queries.iloc[0], records[0])
assert matrix.shape == (3, 38) and np.allclose(matrix[:, 32:36], features[:, 6:10])
assert matrix[1, 36] < matrix[2, 36]  # global semantic order: item4 before item5
assert matrix[2, 37] < matrix[1, 37]  # geo semantic order reverses these items
namespace["legacy_ranker_choice"] = {"alpha": .5}
semantic_scores = [np.array([10., -2., 3.])]
legacy_scores = [np.array([-1., 4., 0.])]
zero_weight = namespace["ensemble_predictions"](records, semantic_scores, legacy_scores, 0, 2)
prior = namespace["blend_predictions"](records, legacy_scores, .5, 2)
assert np.array_equal(zero_weight[0], prior[0]), "Zero semantic weight must preserve the complete old selector"
one_weight = namespace["ensemble_predictions"](records, semantic_scores, legacy_scores, 1, 2)
assert np.array_equal(one_weight[0], ids[namespace["stable_topk"](semantic_scores[0], 2)])
print("Passed: lexical pool preservation, semantic/geo sources, item-ID alignment, semantic negatives, no positive injection, 38-feature shape, full-pool ranks")
print("Passed: ensemble endpoints preserve the old selector or pure semantic learner")
