"""Refit the selected final model and compare actual benchmark margins bitwise.

Reuses deterministic sampled training data, but does not reuse fitted weights.
This checks training determinism separately from cached-model inference replay.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.ranking_deps'),str(ROOT/'.inspection_deps')]
import joblib
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)
CACHE=ROOT/'artifacts/ranking-v5'
manifest=json.loads((CACHE/'manifest.json').read_text())
winner=manifest['selection']['winner'];fp=manifest['fingerprint']
stage='mined' if winner['model']=='mixed_mined' else 'initial'
data=joblib.load(CACHE/f'final_{stage}_{fp}.joblib')
expected=joblib.load(CACHE/manifest['final_model'])
group_mask=np.ones(len(data['sizes']),dtype=bool) if winner['model']!='cold_wide' else data['mode']=='unseen_text'
row_mask=np.repeat(group_mask,data['sizes'])
x,y=data['X'][row_mask],data['y'][row_mask]
sizes=data['sizes'][group_mask]
weights=np.repeat(data['group_weight'][group_mask],sizes)
if winner['model']=='cold_wide':
    weights[:]=1
retrained=lgb.LGBMRanker(**expected.get_params())
started=time.perf_counter()
print('Refitting selected final model:',x.shape,'groups',len(sizes),flush=True)
retrained.fit(x,y,group=sizes,sample_weight=weights)
del data,x,y,weights
import gc
gc.collect()
records,matrices=joblib.load(CACHE/f'benchmark_{fp}.joblib')
comparisons=0
for start in range(0,len(matrices),64):
    block=np.concatenate(matrices[start:start+64])
    a=expected.predict(block,num_iteration=int(winner['trees']))
    b=retrained.predict(block,num_iteration=int(winner['trees']))
    assert np.array_equal(a,b),'Refitting changed benchmark model margins'
    comparisons+=len(a)
assert expected.booster_.model_to_string()==retrained.booster_.model_to_string()
assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()==manifest['answer_sha256']
report={'fitted_weights_reused':False,'deterministic_sampled_data_reused':True,
    'benchmark_pairs_compared':comparisons,'raw_margins_bitwise_equal':True,
    'booster_model_text_equal':True,'original_answer_unchanged':True,
    'elapsed_seconds':round(time.perf_counter()-started,2)}
(CACHE/'refit_reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
