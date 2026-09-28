"""Check unavailable items, unknown texts and allowed cross-context repeats."""
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.inspection_deps'),str(ROOT/'.ranking_deps')]
import numpy as np
import pandas as pd
from quality_trace_v10 import QueryItemEvidence

history=pd.DataFrame({'query_norm':['alpha','alpha','beta'],'item_id':['a','a','b'],
    'search_location_id':[1,2,1],'search_infm_params_text':[' x\n y ','x y','']})
bank=QueryItemEvidence(history)
query=SimpleNamespace(query_norm='alpha',search_location_id=1,search_infm_params_text='x y')
x=bank.features(query,np.array(['a','held']))
assert x.shape==(2,4) and np.isfinite(x).all()
assert np.all(x[1]==0),'An excluded held item must not acquire exact-query evidence'
assert x[0,0]>x[0,1] and x[0,0]==x[0,2],'Cross-location repeats and normalized filters must be counted correctly'
assert x[0,3]==1. and (x[:,3]<=1).all()
query.query_norm='unknown'
assert not bank.features(query,np.array(['a','b'])).any()
print('Exact query-history contracts passed.')
