"""Bounded microcat experiments using the frozen v5 pools and validation contract.

The original answer is untouched until a development-selected candidate has been
trained, checked and reproduced. Auxiliary classifiers use full allowed train
history; every fold removes its own labels and uses the same cold-item policy.
"""
from pathlib import Path
import ast
import sys
import os
import json
import hashlib
import time
import gc

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding='utf-8')
sys.path[:0] = [str(ROOT / p) for p in ['.development', '.ranking_deps', '.inspection_deps', '.semantic_deps']]
V5_SOURCE = ROOT / '.development/ranking_v5.py'
# Notebook-defined cached classes must resolve in this process's __main__.
tree = ast.parse(V5_SOURCE.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                                       and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')]
driver_file = __file__
__file__ = str(V5_SOURCE)
exec(compile(tree, str(V5_SOURCE), 'exec'), globals())
__file__ = driver_file
from microcat_models import (micro_inputs, sparse_inputs, fit_nb, fit_mlp, predict_micro,
                            micro_candidate_features, classification_metrics, MICRO_FEATURE_NAMES)
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)

V5_CACHE = CACHE
V5_FP = fingerprint
MICRO_CACHE = ROOT / 'artifacts/microcat-v6'
MICRO_CACHE.mkdir(exist_ok=True)
MICRO_CONFIG = {'epochs': 35, 'seed': SEED+901, 'classifiers': ['nb', 'mlp'], 'cold_fraction': .9,
                'query_filter_input': True, 'category_input': False, 'encoder_frozen': True}
MICRO_FP = hashlib.sha256(json.dumps({'inputs': input_hashes, 'config': MICRO_CONFIG,
    'base': V5_FP, 'code': hashlib.sha256(Path(driver_file).read_bytes()).hexdigest(),
    'core': hashlib.sha256((ROOT/'.development/microcat_models.py').read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()[:16]
V5_MANIFEST = json.loads((V5_CACHE/'manifest.json').read_text())
V5_WINNER = V5_MANIFEST['selection']['winner']
BASE_ANSWER_HASH = sha256_file(ROOT/'answer.csv')
assert V5_FP == V5_MANIFEST['fingerprint']


def prepare_micro_space():
    inputs = pd.concat([micro_inputs(history_all), micro_inputs(queries)], ignore_index=True)
    inputs = inputs.drop_duplicates('key').sort_values('key').reset_index(drop=True)
    dense_path = MICRO_CACHE/f'input_vectors_{MICRO_FP}.npy'
    if USE_CACHE and dense_path.exists():
        dense = np.load(dense_path, allow_pickle=False)
    else:
        texts = sorted(set(inputs['query']))
        missing = [q for q in texts if q not in semantic_index.query_to_row]
        vectors = semantic_index.encode(missing, 'query: ') if missing else np.empty((0, 384), np.float32)
        lookup = {q: vectors[i] for i, q in enumerate(missing)}
        lookup.update({q: semantic_index.queries[semantic_index.query_to_row[q]] for q in texts if q in semantic_index.query_to_row})
        filters = sorted(set(inputs.filters) - {''})
        filter_vectors = semantic_index.encode(filters, 'query: ')
        filter_lookup = {f: filter_vectors[i] for i, f in enumerate(filters)}
        filter_lookup[''] = np.zeros(384, np.float32)
        dense = np.column_stack([np.stack([lookup[q] for q in inputs['query']]),
                                 np.stack([filter_lookup[f] for f in inputs.filters])]).astype(np.float32)
        save_array(dense, dense_path)
        semantic_index.encoder = semantic_index.tokenizer = None
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    space = {'inputs': inputs, 'input_to_row': dict(zip(inputs.key, range(len(inputs)))),
             'dense': dense, 'sparse': sparse_inputs(inputs),
             'classes': np.sort(history_all.item_microcat_id.unique()).astype(np.int64)}
    assert len(dense) == len(inputs)
    print('Micro input space',len(inputs),'classes',len(space['classes']), 'dense MiB',round(dense.nbytes/2**20,1),flush=True)
    return space


def history_digest(history):
    return hashlib.sha256(pd.util.hash_pandas_object(history[['context_key', 'item_id', 'item_microcat_id']], index=False).values.tobytes()).hexdigest()


def classifier(space, history, tag, kind):
    path = MICRO_CACHE/f'{tag}_{kind}_{MICRO_FP}.joblib'
    digest = history_digest(history)
    if USE_CACHE and path.exists():
        result = joblib.load(path)
        assert result['history_sha256'] == digest
        return result
    started = time.perf_counter()
    model = fit_nb(space, history) if kind == 'nb' else fit_mlp(space, history, MICRO_CONFIG['seed'], DEVICE, MICRO_CONFIG['epochs'])
    model['history_sha256'] = digest
    save_cache(model, path)
    print('Fitted classifier',tag,kind,'history pairs',len(history),'seconds',round(time.perf_counter()-started,1),flush=True)
    return model


def evaluation_history(frame, mode, other_frame):
    # Exactly the same exclusions as v5.evaluation_features.
    truth = labels_from_gold(gold, frame)
    other_truth = labels_from_gold(gold, other_frame)
    other_positive = np.array(sorted({ITEM_IDS[i] for t in other_truth for i in t}), dtype=str)
    random = np.random.default_rng(SEED+812)
    other_cold = set(random.choice(other_positive, int(.9*len(other_positive)), replace=False))
    fit, _, _ = history_for_queries(history_all, frame, truth, ITEM_IDS, mode, .9, SEED+801,
        excluded_texts=other_frame.query_norm, excluded_items=other_cold)
    return fit


def evaluation_micro(space, frame, mode, stage, other_frame):
    fit = evaluation_history(frame, mode, other_frame)
    result = {}
    for kind in MICRO_CONFIG['classifiers']:
        model = classifier(space, fit, f'{stage}_{mode}', kind)
        probabilities = predict_micro(model, space, frame)
        metrics = classification_metrics(probabilities, frame, gold, items.item_microcat_id.to_numpy(), space['classes'])
        result[kind] = {'probabilities': probabilities, 'supported': model['supported'], 'metrics': metrics}
        print('CLASSIFICATION',stage,mode,kind,json.dumps(metrics),flush=True)
        del model
    del fit
    gc.collect()
    save_cache(result, MICRO_CACHE/f'{stage}_{mode}_probabilities_{MICRO_FP}.joblib')
    return result


def baseline_v5(records, matrices, final=False):
    name = V5_MANIFEST['final_model'] if final else f'mixed_mined_{V5_FP}.joblib'
    ranker = joblib.load(V5_CACHE/name)
    return blend_scores(predict_scores(ranker, matrices, int(V5_WINNER['trees'])),
                        old_v4_scores(records, matrices, final=final), float(V5_WINNER['weight']))


def standalone():
    space = prepare_micro_space()
    metrics = {}
    for mode in ['unseen_text', 'held_context']:
        result = evaluation_micro(space, development, mode, 'development', control)
        metrics[mode] = {kind: value['metrics'] for kind, value in result.items()}
    report = {'stage': 'microcat-v6', 'fingerprint': MICRO_FP, 'config': MICRO_CONFIG,
        'classifiers': metrics, 'base_answer_sha256': BASE_ANSWER_HASH,
        'answer_unchanged': sha256_file(ROOT/'answer.csv') == BASE_ANSWER_HASH,
        'control_evaluated': False, 'next': 'OOF candidate features and ranker ablation'}
    assert report['answer_unchanged']
    (MICRO_CACHE/'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STANDALONE COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    standalone()
