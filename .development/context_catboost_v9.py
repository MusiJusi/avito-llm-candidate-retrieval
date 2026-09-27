"""A bounded alternative ranking objective on independently prepared OOF groups.

No new interactions or validation labels enter training. GPU training is not
claimed byte-deterministic: frozen fitted weights are the reproduction artifact.
Prediction is deterministic on CPU and will be verified in a fresh notebook.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import time

CAT_DRIVER = Path(__file__).resolve()
source = CAT_DRIVER.with_name('context_rank_v9.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')]
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())
__file__ = str(CAT_DRIVER)
from catboost import CatBoostRanker, Pool
from context_model_v9 import CatBoostContextModel

PILOT_CACHE = CONTEXT_CACHE
PILOT_FP = CONTEXT_FP
CAT_CONFIG = dict(iterations=400, depth=6, learning_rate=.05, l2_leaf_reg=12,
                  loss_function='YetiRank', task_type='GPU', devices='0',
                  random_seed=SEED, thread_count=6, gpu_ram_part=.65,
                  allow_writing_files=False, verbose=100)
CONTEXT_FP = hashlib.sha256(json.dumps({
    'pilot': PILOT_FP, 'cat_config': CAT_CONFIG,
    'driver': sha256_file(CAT_DRIVER),
    'adapter': sha256_file(CAT_DRIVER.with_name('context_model_v9.py'))}, sort_keys=True).encode()).hexdigest()[:16]
CONTEXT_CACHE = ROOT / 'artifacts/context-catboost-v9'
CONTEXT_CACHE.mkdir(exist_ok=True)
FEATURE_SETS = {'catboost': list(range(70))}
for stage, modes in [('development', ['unseen_text', 'held_context']), ('control', ['unseen_text'])]:
    for mode in modes:
        old = PILOT_CACHE / f'{stage}_{mode}_pools_{PILOT_FP}.joblib'
        new = CONTEXT_CACHE / f'{stage}_{mode}_pools_{CONTEXT_FP}.joblib'
        if old.exists() and not new.exists():os.link(old, new)

def prepare_training(stage='evaluation'):
    if stage != 'evaluation':raise ValueError('Final refit requires the selected export recipe')
    return joblib.load(PILOT_CACHE / f'evaluation_training_{PILOT_FP}.joblib')

def fit_models(data, stage='evaluation', variants=None):
    path = CONTEXT_CACHE / f'{stage}_catboost_{CONTEXT_FP}.joblib'
    if path.exists():return {'catboost': joblib.load(path)}
    group_ids = np.repeat(np.arange(len(data['sizes']), dtype=np.int64), data['sizes'])
    group_weights = np.repeat(data['group_weight'], data['sizes'])
    pool = Pool(data['X'], label=data['y'], group_id=group_ids, group_weight=group_weights)
    started = time.perf_counter()
    print('CatBoost OOF fit', data['X'].shape, CAT_CONFIG, flush=True)
    model = CatBoostRanker(**CAT_CONFIG)
    model.fit(pool)
    wrapped = CatBoostContextModel(model)
    save_cache(wrapped, path)
    path.with_suffix('.json').write_text(json.dumps({
        'seconds': round(time.perf_counter()-started, 2), 'config': CAT_CONFIG,
        'sha256': sha256_file(path), 'frozen_gpu_training': True}, indent=2), encoding='utf-8')
    return {'catboost': wrapped}

if __name__ == '__main__':
    print('CatBoost context experiment', CONTEXT_FP, flush=True)
    comparison()
