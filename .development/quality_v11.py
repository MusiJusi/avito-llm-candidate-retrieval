"""Controlled v11 ablations on the frozen v10 candidate pools.

The input pairs, labels, history exclusions and BGE embeddings are exactly v10.
Only feature blocks and subsequently training examples are changed here.
"""
from pathlib import Path
import argparse
import ast
import gc
import hashlib
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

V11_DRIVER = Path(__file__).resolve()
_upstream = V11_DRIVER.with_name('bge_ranker_v10.py')
_tree = ast.parse(_upstream.read_text(encoding='utf-8'))
_tree.body = [node for node in _tree.body if not (
    isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name)
    and node.test.left.id == '__name__')]
__file__ = str(_upstream)
exec(compile(_tree, str(_upstream), 'exec'), globals())
__file__ = str(V11_DRIVER)


class _Upstream:
    """Access upstream functions that intentionally live in __main__ for joblib."""
    def __getattr__(self, name):
        return globals()[name]

    def __setattr__(self, name, value):
        globals()[name] = value


v10 = _Upstream()
from bge_reference_v11 import MANIFEST as BGE_MANIFEST, OUT as REFERENCE_PATH
from rank_features_v11 import (BgeReference, BGE_RELATIVE_NAMES,
                               ItemMetadata, STRUCTURED_NAMES)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'artifacts/bge-m3-v11'
SEED = v10.SEED + 1811
BASE_WIDTH = 94
REL_WIDTH = BASE_WIDTH + len(BGE_RELATIVE_NAMES)
FULL_WIDTH = REL_WIDTH + len(STRUCTURED_NAMES)
METADATA_WIDTH = BASE_WIDTH + len(STRUCTURED_NAMES)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(structured=False):
    """Load frozen vectors and corpus metadata once per process."""
    v10.install_field_functions()
    v10.BGE_DOCUMENTS, v10.BGE_QUERIES = v10.prepare_bge()
    v10.CONTEXT_VECTORS, _ = v10.contextual_vectors()
    texts = joblib.load(ROOT / BGE_MANIFEST['query_texts']['path'])
    reference = BgeReference(texts, np.load(REFERENCE_PATH, mmap_mode='r'))
    metadata = ItemMetadata(v10.items) if structured else None
    return reference, metadata


def feature_block(query, base, ids, reference, metadata, block):
    if block == 'base':
        return np.empty((len(base), 0), dtype=np.float32)
    filtered = v10.query_filter_text(query.query_norm, query.search_infm_params_text)
    if block == 'metadata':
        return metadata.features(query, ids, base)
    relative = reference.features(query.query_norm, filtered, base)
    if block == 'relative':
        return relative
    return np.column_stack([relative, metadata.features(query, ids, base)]).astype(np.float32)


def matrix_path(stage, block):
    return CACHE / f'{stage}_{block}_X_{v10.BGE_FP}.npy'


def build_matrix(stage, block):
    """Append v11 signals in the exact order of the cached v10 OOF groups."""
    assert block in {'base', 'relative', 'structured', 'metadata'}
    if block == 'base':
        v10.install_field_functions()
        v10.bge_training(stage)
        return
    structured = block in {'structured', 'metadata'}
    path = matrix_path(stage, block)
    report_path = path.with_suffix('.json')
    if path.exists() and report_path.exists():
        print('Existing matrix', path, flush=True)
        return
    reference, metadata = initialize(structured)
    data = v10.bge_training(stage)
    assert data['X'].shape[1] == BASE_WIDTH
    count = len(data['y'])
    width = {'base': BASE_WIDTH, 'relative': REL_WIDTH, 'structured': FULL_WIDTH,
             'metadata': METADATA_WIDTH}[block]
    temporary = path.with_suffix('.npy.partial')
    target = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                       shape=(count, width))
    frame = v10.frame_for_stage(stage)[0].set_index('context_key')
    keys = data['report']['context_keys']
    sizes = data['sizes']
    assert len(keys) == len(sizes)
    ids = None
    if structured:
        def id_groups():
            for _, group_path in v10.ordered_paths(v10.V10_CACHE, stage, v10.V10_FP):
                for group in joblib.load(group_path):
                    yield group[5]
        ids = id_groups()
    start = time.perf_counter()
    offset = 0
    for i, (key, size) in enumerate(zip(keys, sizes)):
        stop = offset + int(size)
        base = data['X'][offset:stop]
        group_ids = next(ids) if structured else None
        if structured:
            assert len(group_ids) == size
        target[offset:stop, :BASE_WIDTH] = base
        target[offset:stop, BASE_WIDTH:] = feature_block(
            frame.loc[key], base, group_ids, reference, metadata, block)
        offset = stop
        if (i + 1) % 2000 == 0:
            print('v11 features', stage, block, i + 1, len(sizes),
                  'seconds', round(time.perf_counter()-start), flush=True)
    assert offset == count
    if structured:
        try:
            next(ids)
            raise AssertionError('More cached groups than v10 training manifest')
        except StopIteration:
            pass
    target.flush()
    del target, data, reference, metadata
    gc.collect()
    temporary.replace(path)
    report_path.write_text(json.dumps({
        'stage': stage, 'block': block, 'rows': count, 'groups': len(sizes),
        'width': width, 'base_fingerprint': v10.BGE_FP,
        'reference': REFERENCE_PATH.relative_to(ROOT).as_posix(),
        'feature_names': (BGE_RELATIVE_NAMES if block != 'metadata' else [])
            + (STRUCTURED_NAMES if structured else []),
        'elapsed_seconds': round(time.perf_counter()-start, 2),
        'source_sha256': {p: file_hash(ROOT/'.development'/p) for p in
                          ['quality_v11.py', 'rank_features_v11.py', 'bge_reference_v11.py']},
    }, indent=2), encoding='utf-8')
    print('Matrix ready', path, flush=True)


def model_path(stage, block, recipe='normal'):
    return CACHE / f'{stage}_{block}_{recipe}_{v10.BGE_FP}.joblib'


def fit(stage, block, recipe='normal'):
    path = model_path(stage, block, recipe)
    if path.exists():
        print('Existing ranker', path, flush=True)
        return joblib.load(path)
    build_matrix(stage, block)
    v10.install_field_functions()
    data = v10.bge_training(stage)
    x = np.load(matrix_path(stage, block), mmap_mode='r')
    assert x.shape[0] == len(data['y'])
    options = {'normal': (600, 63, 150, 55),
               'boundary': (600, 63, 150, 70),
               'no_geo': (600, 63, 150, 55)}
    trees, leaves, min_child, truncation = options[recipe]
    if recipe == 'no_geo':
        assert block == 'relative'
    ignored = [102, 103] if recipe == 'no_geo' else []
    model = lgb.LGBMRanker(n_estimators=trees, num_leaves=leaves,
        learning_rate=.04, reg_lambda=15, max_bin=127,
        min_child_samples=min_child, random_state=SEED, n_jobs=8,
        verbosity=-1, deterministic=True, force_col_wise=True,
        lambdarank_truncation_level=truncation, label_gain=[0, 1],
        ignore_column=ignored)
    start = time.perf_counter()
    print('Fit v11', stage, block, recipe, x.shape, flush=True)
    model.fit(x, data['y'], group=data['sizes'],
              sample_weight=np.repeat(data['group_weight'], data['sizes']))
    joblib.dump(model, path, compress=3)
    path.with_suffix('.json').write_text(json.dumps({
        'stage': stage, 'block': block, 'recipe': recipe,
        'rows': len(data['y']), 'groups': len(data['sizes']),
        'seconds': round(time.perf_counter()-start, 2),
        'ignored_columns': ignored,
        'model_sha256': file_hash(path), 'features': x.shape[1],
    }, indent=2), encoding='utf-8')
    print('Fit ready', path, 'seconds', round(time.perf_counter()-start), flush=True)
    return model


def evaluation_pool(stage, mode, block, reference, metadata):
    """Produce full-pool features once; never recompute ranks after sampling."""
    path = CACHE / f'{stage}_{mode}_{block}_pool_{v10.BGE_FP}.joblib'
    if path.exists():
        return joblib.load(path)
    frame, records, base, _, known = v10.v9_pool(stage, mode)
    history = v10.evaluation_history(frame, mode,
        v10.control if stage == 'development' else v10.development)
    v10.QUALITY_BANK = v10.QualityEvidence(history)
    del history
    gc.collect()
    x = v10.extra_matrices(frame, records, base)
    del base
    for q, values in zip(frame.itertuples(index=False), x):
        if float(q.search_category) == 0:
            values[:,70] = 1.
    x = [np.column_stack([values, v10.field_features(q, ids, values)]).astype(np.float32)
         for q, (ids, _), values in zip(frame.itertuples(index=False), records, x)]
    bge = [v10.bge_features(q, ids, values)
           for q, (ids, _), values in zip(frame.itertuples(index=False), records, x)]
    x = [np.column_stack([values, extra]).astype(np.float32)
         for values, extra in zip(x, bge)]
    assert x[0].shape[1] == BASE_WIDTH
    selected = [np.column_stack([values, feature_block(
        q, values, ids, reference, metadata, block)]).astype(np.float32)
        for q, (ids, _), values in zip(frame.itertuples(index=False), records, x)]
    assert selected[0].shape[1] == {'base': BASE_WIDTH, 'relative': REL_WIDTH,
        'structured': FULL_WIDTH, 'metadata': METADATA_WIDTH}[block]
    payload = {'matrix': selected, 'known': known}
    joblib.dump(payload, path, compress=1)
    print('Evaluation pool ready', stage, mode, block, flush=True)
    return payload


def evaluate(block, recipe='normal'):
    model = fit_hard(block) if recipe == 'hard' else fit('evaluation', block, recipe)
    original = joblib.load(v10.bge_model_path('evaluation', 'deep'))
    reference, metadata = initialize(block in {'structured', 'metadata'})
    results = {}
    baseline_results = {}
    blend_results = {}
    for mode in ['unseen_text', 'held_context']:
        pool = evaluation_pool('development', mode, block, reference, metadata)
        frame, records, _, _, _ = v10.v9_pool('development', mode)
        x = pool['matrix']
        pred = v10.predict_scores(model, x, 600)
        old = v10.predict_scores(original, [a[:, :BASE_WIDTH] for a in x], 600)
        truth = v10.labels_from_gold(v10.gold, frame)
        results[mode] = v10.per_query_recall(v10.top50(records, pred), truth)
        baseline_results[mode] = v10.per_query_recall(v10.top50(records, old), truth)
        blend_results[mode] = {weight: v10.per_query_recall(v10.top50(
            records, v10.blend_scores(pred, old, weight)), truth)
            for weight in [.25, .5, .75, 1.]}
        print('Mode', mode, 'v10', float(baseline_results[mode].mean()),
              'v11', float(results[mode].mean()), flush=True)
    known = pool['known']
    rows = {
        'baseline': v10.matched_metrics(baseline_results['unseen_text'],
             baseline_results['held_context'], known),
        'candidate': v10.matched_metrics(results['unseen_text'],
             results['held_context'], known),
    }
    rows['archived_baseline_matched_recall50'] = .9597256528572226
    rows['baseline_reconstruction_delta'] = (
        rows['baseline']['matched_recall50'] - rows['archived_baseline_matched_recall50'])
    rows['blends'] = {str(weight): v10.matched_metrics(
        blend_results['unseen_text'][weight],
        blend_results['held_context'][weight], known)
        for weight in [.25, .5, .75, 1.]}
    for mode in results:
        rows[mode] = {'improved_queries': int((results[mode] > baseline_results[mode]).sum()),
                      'worsened_queries': int((results[mode] < baseline_results[mode]).sum())}
    path = CACHE / f'development_{block}_{recipe}_{v10.BGE_FP}.json'
    path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('v11 development', json.dumps(rows), flush=True)


def evaluate_joint():
    """Compare a fixed, small RRF grid of complementary v11 rankers."""
    paths = {
        'v10': v10.bge_model_path('evaluation', 'deep'),
        'metadata': model_path('evaluation', 'metadata'),
        'hard_base': model_path('evaluation', 'hard_base'),
        'hard_metadata': model_path('evaluation', 'hard_metadata'),
    }
    assert all(path.exists() for path in paths.values())
    models = {name: joblib.load(path) for name, path in paths.items()}
    recipes = {
        'metadata_half': {'v10': .5, 'metadata': .5},
        'hard_metadata_half': {'v10': .5, 'hard_metadata': .5},
        'three_way_base': {'v10': .5, 'metadata': .25, 'hard_base': .25},
        'three_way_meta': {'v10': .5, 'metadata': .25, 'hard_metadata': .25},
        'three_way_hard': {'v10': .5, 'hard_base': .25, 'hard_metadata': .25},
        'four_way': {'v10': .4, 'metadata': .2, 'hard_base': .2,
                     'hard_metadata': .2},
    }
    values = {name: {} for name in recipes}
    baseline = {}
    for mode in ['unseen_text', 'held_context']:
        pool = evaluation_pool('development', mode, 'metadata', None, None)
        frame, records, _, _, known = v10.v9_pool('development', mode)
        matrices = pool['matrix']
        predictions = {
            name: v10.predict_scores(model, [x[:, :BASE_WIDTH] for x in matrices]
                if name in {'v10', 'hard_base'} else matrices, 600)
            for name, model in models.items()
        }
        truth = v10.labels_from_gold(v10.gold, frame)
        baseline[mode] = v10.per_query_recall(
            v10.top50(records, predictions['v10']), truth)
        reciprocal = {name: [1/(60+v10.rankdata(-score, method='min'))
                             for score in scores]
                      for name, scores in predictions.items()}
        for name, weights in recipes.items():
            combined = [sum(weight*reciprocal[key][i] for key, weight in weights.items())
                        for i in range(len(records))]
            values[name][mode] = v10.per_query_recall(
                v10.top50(records, combined), truth)
        print('Joint predictions ready', mode, flush=True)
    rows = {'baseline': v10.matched_metrics(baseline['unseen_text'],
                                            baseline['held_context'], known),
            'recipes': {name: dict(weights=recipes[name],
                      **v10.matched_metrics(part['unseen_text'],
                                             part['held_context'], known))
                        for name, part in values.items()}}
    path = CACHE / f'development_joint_{v10.BGE_FP}.json'
    path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('v11 joint', json.dumps(rows), flush=True)


def _text_fold(text):
    # The same query text has the same fold in both historical regimes.
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16) % 3


def mine_hard_examples(block, stage='evaluation'):
    """Use text-held teachers to emphasize previously confusing negatives.

    The teacher of a group never saw labels for that query text. Retrieval and
    embeddings are the frozen v10 OOF producers; no hidden labels are used.
    """
    assert block in {'base', 'relative', 'structured', 'metadata'}
    output = matrix_path(stage, f'hard_{block}')
    report_path = output.with_suffix('.json')
    if output.exists() and report_path.exists():
        print('Existing hard-example matrix', output, flush=True)
        return
    build_matrix(stage, block)
    v10.install_field_functions()
    data = v10.bge_training(stage)
    original = (data['X'] if block == 'base' else
                np.load(matrix_path(stage, block), mmap_mode='r'))
    sizes = np.asarray(data['sizes'], dtype=np.int32)
    starts = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    folds = np.asarray([_text_fold(t) for t in data['report']['query_texts']],
                       dtype=np.int8)
    assert len(folds) == len(sizes) and starts[-1] == len(data['y'])
    base_scores = CACHE / f'{stage}_base_textheld_scores_{v10.BGE_FP}.npy'
    teacher_block = 'base' if block != 'base' and base_scores.exists() else block
    scores_path = CACHE / f'{stage}_{teacher_block}_textheld_scores_{v10.BGE_FP}.npy'
    if scores_path.exists():
        scores = np.load(scores_path, mmap_mode='r')
    else:
        temp_scores = scores_path.with_suffix('.npy.partial')
        scores = np.lib.format.open_memmap(temp_scores, mode='w+',
                                           dtype=np.float32, shape=(int(starts[-1]),))
        for held in range(3):
            train_groups = np.flatnonzero(folds != held)
            held_groups = np.flatnonzero(folds == held)
            texts = data['report']['query_texts']
            assert not ({texts[i] for i in train_groups} &
                        {texts[i] for i in held_groups})
            train_sizes = sizes[train_groups]
            total = int(train_sizes.sum())
            temp_x = CACHE / f'temporary_{stage}_teacher_fold_{held}_X.npy'
            temp_y = CACHE / f'temporary_{stage}_teacher_fold_{held}_y.npy'
            x = np.lib.format.open_memmap(temp_x, mode='w+', dtype=np.float32,
                                           shape=(total, original.shape[1]))
            y = np.lib.format.open_memmap(temp_y, mode='w+', dtype=np.uint8,
                                           shape=(total,))
            offset = 0
            for group in train_groups:
                a, b = starts[group:group+2]
                stop = offset + (b-a)
                x[offset:stop] = original[a:b]
                y[offset:stop] = data['y'][a:b]
                offset = stop
            x.flush(); y.flush()
            assert offset == total
            teacher = lgb.LGBMRanker(n_estimators=250, num_leaves=31,
                learning_rate=.05, reg_lambda=20, max_bin=127,
                min_child_samples=100, random_state=SEED+held,
                n_jobs=8, verbosity=-1, deterministic=True,
                force_col_wise=True, lambdarank_truncation_level=75,
                label_gain=[0, 1])
            started = time.perf_counter()
            print('Fit text-held v11 teacher', held, 'rows', total,
                  'groups', len(train_groups), flush=True)
            teacher.fit(x, y, group=train_sizes,
                sample_weight=np.repeat(data['group_weight'][train_groups], train_sizes))
            for group in held_groups:
                a, b = starts[group:group+2]
                scores[a:b] = teacher.predict(original[a:b], num_iteration=250)
            print('Teacher ready', held, 'seconds', round(time.perf_counter()-started),
                  flush=True)
            del x, y, teacher
            gc.collect()
            temp_x.unlink(); temp_y.unlink()
        scores.flush(); del scores
        temp_scores.replace(scores_path)
        scores = np.load(scores_path, mmap_mode='r')
    assert np.isfinite(scores).all()
    # A second pass materializes only selected rows. The group boundaries are
    # preserved, and all retrieved positives remain in their original group.
    chosen = []
    target_sizes = []
    hard_stats = []
    for group, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        y = data['y'][a:b]
        positives = np.flatnonzero(y)
        negatives = np.flatnonzero(~y.astype(bool))
        order = negatives[np.lexsort((negatives, -scores[a:b][negatives]))]
        hard = order[:min(400, len(order))]
        tail = np.setdiff1d(negatives, hard, assume_unique=True)
        rng = np.random.default_rng(int(hashlib.sha256(
            ('v11-hard:'+data['report']['context_keys'][group]).encode()).hexdigest()[:8], 16))
        random_tail = rng.choice(tail, size=min(96, len(tail)), replace=False)
        keep = np.unique(np.r_[positives, hard, random_tail])
        assert len(keep) > len(positives) and y[keep].sum() == y.sum()
        chosen.append(keep)
        target_sizes.append(len(keep))
        hard_stats.append(len(hard))
    total = sum(target_sizes)
    tmp = output.with_suffix('.npy.partial')
    target = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32,
                                       shape=(total, original.shape[1]))
    y_output = output.with_name(output.stem.replace('_X_', '_y_')+'.npy')
    y_tmp = y_output.with_suffix('.npy.partial')
    target_y = np.lib.format.open_memmap(y_tmp, mode='w+', dtype=np.uint8,
                                         shape=(total,))
    offset = 0
    for (a, b), keep in zip(zip(starts[:-1], starts[1:]), chosen):
        stop = offset + len(keep)
        target[offset:stop] = original[a:b][keep]
        target_y[offset:stop] = data['y'][a:b][keep]
        offset = stop
    assert offset == total
    target.flush(); target_y.flush()
    del target, target_y
    tmp.replace(output); y_tmp.replace(y_output)
    report_path.write_text(json.dumps({
        'block': block, 'stage': stage,
        'source_training': str((v10.BGE_CACHE/f'{stage}_X_{v10.BGE_FP}.npy'
            if block == 'base' else matrix_path(stage,block)).relative_to(ROOT)),
        'groups': len(sizes), 'rows': total, 'original_rows': int(starts[-1]),
        'positive_count': int(np.asarray(data['y']).sum()),
        'mean_group_size': float(np.mean(target_sizes)),
        'teacher_folds': 3, 'teacher_trees': 250,
        'teacher_feature_block': teacher_block,
        'text_fold_isolation': True, 'hard_negatives': 400, 'random_tail': 96,
        'scores_file': scores_path.relative_to(ROOT).as_posix(),
        'y_file': y_output.relative_to(ROOT).as_posix(),
        'sizes': target_sizes,
    }, indent=2), encoding='utf-8')
    print('Hard-example matrix ready', output, 'rows', total, flush=True)


def fit_hard(block, stage='evaluation'):
    path = model_path(stage, f'hard_{block}')
    if path.exists():return joblib.load(path)
    mine_hard_examples(block, stage)
    v10.install_field_functions()
    data = v10.bge_training(stage)
    report = json.loads(matrix_path(stage, f'hard_{block}').with_suffix('.json').read_text())
    x = np.load(matrix_path(stage, f'hard_{block}'), mmap_mode='r')
    y = np.load(ROOT / report['y_file'], mmap_mode='r')
    sizes = np.asarray(report['sizes'], np.int32)
    assert len(sizes) == len(data['sizes']) and sizes.sum() == len(y)
    model = lgb.LGBMRanker(n_estimators=600, num_leaves=63,
        learning_rate=.04, reg_lambda=15, max_bin=127,
        min_child_samples=150, random_state=SEED,
        n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True,
        lambdarank_truncation_level=55, label_gain=[0, 1])
    started = time.perf_counter()
    print('Fit v11 hard examples', block, x.shape, flush=True)
    model.fit(x, y, group=sizes,
              sample_weight=np.repeat(data['group_weight'], sizes))
    joblib.dump(model, path, compress=3)
    path.with_suffix('.json').write_text(json.dumps({
        'block': block, 'rows': len(y), 'groups': len(sizes),
        'seconds': round(time.perf_counter()-started, 2),
        'model_sha256': file_hash(path)}, indent=2), encoding='utf-8')
    return model


def category_gate_audit():
    """Check whether the permitted category can safely free top-50 slots."""
    model = joblib.load(v10.bge_model_path('evaluation', 'deep'))
    rows = {}
    for mode in ['unseen_text', 'held_context']:
        pool = evaluation_pool('development', mode, 'relative', None, None)
        frame, records, _, _, known = v10.v9_pool('development', mode)
        scores = v10.predict_scores(model,
            [x[:, :BASE_WIDTH] for x in pool['matrix']], 600)
        gated = []
        excluded = 0
        for q, (ids, _), values in zip(frame.itertuples(index=False),
                                       records, scores):
            copy = values.copy()
            if float(q.search_category) > 0:
                incompatible = v10.ITEM_CATEGORIES[ids] != float(q.search_category)
                excluded += int(incompatible.sum())
                copy[incompatible] = -1e9
            gated.append(copy)
        truth = v10.labels_from_gold(v10.gold, frame)
        before = v10.per_query_recall(v10.top50(records, scores), truth)
        after = v10.per_query_recall(v10.top50(records, gated), truth)
        rows[mode] = {'before': float(before.mean()), 'after': float(after.mean()),
            'improved': int((after > before).sum()),
            'worsened': int((after < before).sum()),
            'excluded_candidates': excluded}
    path = CACHE/'category_gate_development.json'
    path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('Category gate audit', json.dumps(rows), flush=True)


def freeze_benchmark_reference():
    """Package only the corpus score references needed by benchmark queries."""
    path = CACHE/f'benchmark_reference_{v10.BGE_FP}.joblib'
    if path.exists():
        print('Existing benchmark reference', path, flush=True)
        return path
    texts = joblib.load(ROOT/BGE_MANIFEST['query_texts']['path'])
    lookup = {text: index for index, text in enumerate(texts)}
    needed = sorted(set(v10.queries.query_norm) | {
        v10.query_filter_text(q, f) for q, f in v10.queries[
            ['query_norm', 'search_infm_params_text']].itertuples(index=False, name=None)})
    values = np.load(REFERENCE_PATH, mmap_mode='r')
    assert all(text in lookup for text in needed)
    frozen = np.stack([values[lookup[text]] for text in needed])
    joblib.dump({'texts': needed, 'scores': frozen,
                 'source_fingerprint': v10.BGE_FP}, path, compress=3)
    print('Benchmark reference ready', path, 'texts', len(needed), flush=True)
    return path


def recheck_original_baseline():
    """Recompute the archived v10 comparison through its original evaluator."""
    v10.install_field_functions()
    v10.BGE_DOCUMENTS, v10.BGE_QUERIES = v10.prepare_bge()
    v10.CONTEXT_VECTORS, _ = v10.contextual_vectors()
    model = joblib.load(v10.bge_model_path('evaluation', 'deep'))
    a, _ = v10.evaluate_bge('development', 'unseen_text', {'deep': model})
    b, known = v10.evaluate_bge('development', 'held_context', {'deep': model})
    measured = v10.matched_metrics(a[('ranker','deep',1.)],
                                   b[('ranker','deep',1.)], known)
    report = {'original_evaluator': measured,
              'archived_selection': json.loads((v10.BGE_CACHE/'selection.json').read_text())['winner']}
    (CACHE/'rechecked_v10_baseline.json').write_text(json.dumps(report, indent=2),
                                                    encoding='utf-8')
    print('Rechecked v10 baseline', json.dumps(report), flush=True)


def check_control(block, recipe, weight):
    """Inspect the reused control once after development fixes the variant."""
    assert block in {'base', 'relative', 'structured', 'metadata'}
    assert weight in {0.25, 0.5, 0.75, 1.0}
    candidate = fit_hard(block) if recipe == 'hard' else fit('evaluation', block, recipe)
    original = joblib.load(v10.bge_model_path('evaluation', 'deep'))
    reference, metadata = initialize(block in {'structured','metadata'})
    pool = evaluation_pool('control', 'unseen_text', block, reference, metadata)
    frame, records, _, _, _ = v10.v9_pool('control','unseen_text')
    x = pool['matrix']
    old = v10.predict_scores(original, [a[:,:BASE_WIDTH] for a in x], 600)
    new = v10.predict_scores(candidate, x, 600)
    truth = v10.labels_from_gold(v10.gold, frame)
    before = v10.per_query_recall(v10.top50(records, old), truth)
    after = v10.per_query_recall(v10.top50(records,
        v10.blend_scores(new, old, weight)), truth)
    report = {'block': block, 'recipe': recipe, 'weight': weight,
        'baseline_recall50': float(before.mean()),
        'candidate_recall50': float(after.mean()),
        'improved_queries': int((after > before).sum()),
        'worsened_queries': int((after < before).sum()),
        'control_reused': True, 'used_for_hyperparameter_selection': False}
    path = CACHE/f'control_{block}_{recipe}_{v10.BGE_FP}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('v11 control', json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['matrix', 'fit', 'evaluate', 'hardmine', 'hardfit',
                                           'categoryaudit', 'freeze', 'recheckbaseline',
                                           'control', 'joint'])
    parser.add_argument('--stage', choices=['evaluation', 'final'], default='evaluation')
    parser.add_argument('--block', choices=['base', 'relative', 'structured', 'metadata'], default='relative')
    parser.add_argument('--recipe', choices=['normal', 'boundary', 'no_geo', 'hard'], default='normal')
    parser.add_argument('--weight', type=float, default=0.5)
    args = parser.parse_args()
    if args.action == 'matrix': build_matrix(args.stage, args.block)
    elif args.action == 'fit': fit(args.stage, args.block, args.recipe)
    elif args.action == 'hardmine': mine_hard_examples(args.block, args.stage)
    elif args.action == 'hardfit': fit_hard(args.block, args.stage)
    elif args.action == 'categoryaudit': category_gate_audit()
    elif args.action == 'freeze': freeze_benchmark_reference()
    elif args.action == 'recheckbaseline': recheck_original_baseline()
    elif args.action == 'control': check_control(args.block, args.recipe, args.weight)
    elif args.action == 'joint': evaluate_joint()
    else: evaluate(args.block, args.recipe)


if __name__ == '__main__':
    main()
