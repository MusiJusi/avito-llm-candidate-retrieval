"""Fast v12 experiments on frozen v11 validation pools.

This module deliberately reuses the candidates, labels, features and trained
rankers from v11.  It searches only a small, predefined family of rank-fusion
recipes and reports paired changes against the submitted v11 recipe.
"""
from pathlib import Path
import argparse
import ast
import gc
import hashlib
import json

import joblib
import numpy as np

V12_DRIVER = Path(__file__).resolve()
_source = V12_DRIVER.with_name("quality_v11.py")
_tree = ast.parse(_source.read_text(encoding="utf-8"))
_tree.body = [node for node in _tree.body if not (
    isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")]
__file__ = str(_source)
exec(compile(_tree, str(_source), "exec"), globals())
__file__ = str(V12_DRIVER)


class _V11:
    def __getattr__(self, name):
        return globals()[name]


v11 = _V11()


ROOT = Path(__file__).resolve().parents[1]
V12_CACHE = ROOT / "artifacts/quality-v12"
V12_CACHE.mkdir(parents=True, exist_ok=True)
FP = v11.v10.BGE_FP
MODES = ("unseen_text", "held_context")


def score_path(mode):
    return V12_CACHE / f"development_{mode}_ranks_{FP}.joblib"


def build_score_cache():
    """Predict five existing models once and cache full-pool ranks."""
    paths = {
        "v10": v11.v10.bge_model_path("evaluation", "deep"),
        "metadata": v11.model_path("evaluation", "metadata"),
        "relative": v11.model_path("evaluation", "relative"),
        "hard_base": v11.model_path("evaluation", "hard_base"),
        "hard_metadata": v11.model_path("evaluation", "hard_metadata"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    assert not missing, missing
    models = {name: joblib.load(path) for name, path in paths.items()}
    for mode in MODES:
        path = score_path(mode)
        if path.exists():
            print("Existing score cache", path, flush=True)
            continue
        metadata_pool = v11.evaluation_pool("development", mode, "metadata", None, None)
        relative_pool = v11.evaluation_pool("development", mode, "relative", None, None)
        frame, records, _, _, known = v11.v10.v9_pool("development", mode)
        metadata = metadata_pool["matrix"]
        relative = relative_pool["matrix"]
        assert len(metadata) == len(relative) == len(records)
        matrices = {
            "v10": [x[:, :v11.BASE_WIDTH] for x in metadata],
            "metadata": metadata,
            "relative": relative,
            "hard_base": [x[:, :v11.BASE_WIDTH] for x in metadata],
            "hard_metadata": metadata,
        }
        ranks = {}
        for name, model in models.items():
            scores = v11.v10.predict_scores(model, matrices[name], 600)
            ranks[name] = [v11.v10.rankdata(-score, method="min").astype(np.float32)
                           for score in scores]
            print("Ranks ready", mode, name, flush=True)
            del scores
            gc.collect()
        truth = v11.v10.labels_from_gold(v11.v10.gold, frame)
        payload = {
            "ranks": ranks,
            "known": np.asarray(known, dtype=bool),
            "truth": truth,
            "context_keys": frame.context_key.astype(str).tolist(),
            "record_lengths": [len(row[0]) for row in records],
        }
        joblib.dump(payload, path, compress=1)
        print("Saved", path, flush=True)
        del metadata_pool, relative_pool, metadata, relative, ranks, payload, records
        gc.collect()


def recipes(include_structured=False):
    """A compact grid fixed before reading its validation scores."""
    values = []
    for constant in (30, 45, 60, 80, 110):
        for metadata_weight in (.35, .40, .45, .50, .55, .60, .65):
            values.append((f"two_c{constant}_m{metadata_weight:.2f}", constant,
                           {"v10": 1 - metadata_weight,
                            "metadata": metadata_weight}))
    for constant in (45, 60, 80):
        for relative_weight in (.05, .10, .15, .20):
            remainder = 1 - relative_weight
            for metadata_share in (.45, .50, .55):
                values.append((
                    f"relative_c{constant}_r{relative_weight:.2f}_m{metadata_share:.2f}",
                    constant,
                    {"v10": remainder * (1 - metadata_share),
                     "metadata": remainder * metadata_share,
                     "relative": relative_weight},
                ))
    if include_structured:
        # First compare the joint model directly with v10, then use it as a
        # conservative correction to the submitted and relative ensembles.
        for constant in (45, 60, 80):
            for joint_weight in (.30, .40, .50, .60, .70):
                values.append((
                    f"joint_c{constant}_j{joint_weight:.2f}", constant,
                    {"v10": 1 - joint_weight, "structured": joint_weight},
                ))
            for joint_weight in (.10, .20, .30):
                remainder = 1 - joint_weight
                values.append((
                    f"submitted_joint_c{constant}_j{joint_weight:.2f}", constant,
                    {"v10": remainder / 2, "metadata": remainder / 2,
                     "structured": joint_weight},
                ))
                values.append((
                    f"relative_joint_c{constant}_j{joint_weight:.2f}", constant,
                    {"v10": .36 * remainder, "metadata": .44 * remainder,
                     "relative": .20 * remainder, "structured": joint_weight},
                ))
    # The hard-mined models were weaker alone, so only small additions are tested.
    for constant in (45, 60, 80):
        for hard_name in ("hard_base", "hard_metadata"):
            for hard_weight in (.05, .10, .15):
                remainder = 1 - hard_weight
                values.append((
                    f"{hard_name}_c{constant}_h{hard_weight:.2f}", constant,
                    {"v10": remainder / 2, "metadata": remainder / 2,
                     hard_name: hard_weight},
                ))
    # Deduplicate the submitted recipe, which appears once in each constant grid.
    unique = {}
    for name, constant, weights in values:
        key = (constant, tuple(sorted((k, round(v, 8)) for k, v in weights.items())))
        unique[key] = (name, constant, weights)
    return list(unique.values())


def predictions(records, rank_rows, constant, weights):
    result = []
    for row, (ids, _) in enumerate(records):
        fused = np.zeros(len(ids), dtype=np.float64)
        for name, weight in weights.items():
            fused += weight / (constant + rank_rows[name][row])
        result.append(ids[v11.v10.stable_topk(fused, 50)])
    return result


def fold_summary(keys, candidate, baseline):
    folds = np.asarray([int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 5
                        for key in keys], dtype=np.int8)
    delta = candidate - baseline
    return [float(delta[folds == fold].mean()) for fold in range(5)]


def add_structured_ranks():
    """Score the 115-feature model in bounded batches without another pool file."""
    model_path = v11.model_path("evaluation", "structured")
    assert model_path.exists(), model_path
    model = joblib.load(model_path)
    for mode in MODES:
        path = score_path(mode)
        bundle = joblib.load(path)
        if "structured" in bundle["ranks"]:
            print("Existing structured ranks", mode, flush=True)
            continue
        metadata = v11.evaluation_pool("development", mode, "metadata", None, None)["matrix"]
        relative = v11.evaluation_pool("development", mode, "relative", None, None)["matrix"]
        output = []
        for start in range(0, len(metadata), 32):
            block = [np.column_stack([rel, meta[:, v11.BASE_WIDTH:]]).astype(np.float32)
                     for rel, meta in zip(relative[start:start+32],
                                          metadata[start:start+32])]
            scores = model.predict(np.concatenate(block), num_iteration=600)
            pieces = np.split(scores, np.cumsum([len(x) for x in block])[:-1])
            output.extend(v11.v10.rankdata(-score, method="min").astype(np.float32)
                          for score in pieces)
        bundle["ranks"]["structured"] = output
        joblib.dump(bundle, path, compress=1)
        print("Structured ranks ready", mode, flush=True)
        del metadata, relative, output, bundle
        gc.collect()


def sweep(include_structured=False):
    build_score_cache()
    if include_structured:
        add_structured_ranks()
    bundles = {mode: joblib.load(score_path(mode)) for mode in MODES}
    frames = {}
    records = {}
    for mode in MODES:
        frames[mode], records[mode], _, _, _ = v11.v10.v9_pool("development", mode)
    submitted = {}
    for mode in MODES:
        bundle = bundles[mode]
        pred = predictions(records[mode], bundle["ranks"], 60,
                           {"v10": .5, "metadata": .5})
        submitted[mode] = v11.v10.per_query_recall(pred, bundle["truth"])
    baseline_metrics = v11.v10.matched_metrics(
        submitted["unseen_text"], submitted["held_context"],
        bundles["held_context"]["known"])
    rows = []
    candidates = recipes(include_structured=include_structured)
    for number, (name, constant, weights) in enumerate(candidates, 1):
        recalls = {}
        folds = {}
        improved = {}
        worsened = {}
        for mode in MODES:
            bundle = bundles[mode]
            pred = predictions(records[mode], bundle["ranks"], constant, weights)
            recalls[mode] = v11.v10.per_query_recall(pred, bundle["truth"])
            folds[mode] = fold_summary(bundle["context_keys"], recalls[mode],
                                       submitted[mode])
            improved[mode] = int((recalls[mode] > submitted[mode]).sum())
            worsened[mode] = int((recalls[mode] < submitted[mode]).sum())
        metrics = v11.v10.matched_metrics(
            recalls["unseen_text"], recalls["held_context"],
            bundles["held_context"]["known"])
        rows.append({
            "name": name,
            "constant": constant,
            "weights": weights,
            **metrics,
            "delta_matched": metrics["matched_recall50"] - baseline_metrics["matched_recall50"],
            "delta_unseen": metrics["unseen_macro"] - baseline_metrics["unseen_macro"],
            "delta_held": metrics["held_macro"] - baseline_metrics["held_macro"],
            "improved": improved,
            "worsened": worsened,
            "fold_delta": folds,
            "positive_fold_count": sum(value > 0 for part in folds.values() for value in part),
            "negative_fold_count": sum(value < 0 for part in folds.values() for value in part),
        })
        if number % 20 == 0:
            print("Evaluated recipes", number, len(candidates), flush=True)
    rows.sort(key=lambda row: row["matched_recall50"], reverse=True)
    report = {
        "submitted_v11": baseline_metrics,
        "recipe_count": len(rows),
        "includes_structured": include_structured,
        "selection_labels": "development labels derived from train only",
        "top": rows[:30],
        "all": rows,
    }
    path = V12_CACHE / ("structured_ensemble_sweep.json" if include_structured
                        else "ensemble_sweep.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Submitted", json.dumps(baseline_metrics), flush=True)
    print("Best", json.dumps(rows[0], ensure_ascii=False), flush=True)
    print("Saved", path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["scores", "sweep", "structured"])
    args = parser.parse_args()
    if args.action == "scores":
        build_score_cache()
    elif args.action == "sweep":
        sweep()
    else:
        sweep(include_structured=True)


if __name__ == "__main__":
    main()
