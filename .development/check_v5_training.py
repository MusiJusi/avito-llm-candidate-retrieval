"""Exercise the actual v5 sampling and feature functions without fitting big models."""
import ast
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.inspection_deps'))
import numpy as np
import pandas as pd
from types import SimpleNamespace
source=(ROOT/'.development/ranking_v5.py').read_text(encoding='utf-8')
functions={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
nb=json.loads((ROOT/'Avito.ipynb').read_text(encoding='utf-8'))
stable=None
for cell in nb['cells']:
    if cell['cell_type']=='code':
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef) and node.name=='stable_topk':
                stable=node
ns={'np':np,'V5_CONFIG':{'hard_lexical':2,'hard_geo_semantic':2,'hard_semantic':2,
                        'random_tail':2,'mined_hard':2},'best_config':{},
    'score_candidates':lambda x,c:x[:,0]}
for node in [stable,functions['sample_v5_group'],functions['v5_features']]:
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual v5 function>','exec'),ns)
ids=np.arange(20)
base=np.zeros((20,10),dtype=np.float32)
base[:,0]=np.arange(20);base[:,6]=np.arange(20)[::-1];base[:,7]=np.arange(20)
features=np.zeros((20,45),dtype=np.float32)
features[:,0]=np.arange(20)
class Miner:
    def predict(self,x,num_iteration):
        assert num_iteration==200
        return -x[:,0]
for miner in [None,Miner()]:
    chosen,y=ns['sample_v5_group'](ids,base,features,{3,8},np.random.default_rng(42),miner)
    assert {3,8} <= set(ids[chosen]) and int(y[chosen].sum())==2
    assert len(chosen)==len(set(chosen))
    repeat,_=ns['sample_v5_group'](ids,base,features,{3,8},np.random.default_rng(42),miner)
    assert np.array_equal(chosen,repeat)
missing,_=ns['sample_v5_group'](ids,base,features,{999},np.random.default_rng(42))
assert len(missing)==0,'Missing positives were injected into candidates'
ns.update({'V5_FEATURE_NAMES':list(range(45)),'ITEM_LOCS':np.array([1,2,1]),
    'centers':pd.DataFrame([[55.,37.]],index=[1]),
    'rank_semantic_features':lambda q,r:np.tile(np.arange(38),(len(r[0]),1))})
history=SimpleNamespace(text_to_row={'alpha':0},query_counts={'alpha':5},
    entropy=np.array([.2]),peak=np.array([.8]),location_probability={1:{1:.7,2:.3}},
    location_counts={1:10})
q=SimpleNamespace(query_norm='alpha',search_location_id=1)
matrix=ns['v5_features'](q,(np.arange(3),None),history)
assert matrix.shape==(3,45) and np.array_equal(matrix[:,:38],np.tile(np.arange(38),(3,1)))
assert np.all(matrix[:,38]==1) and np.allclose(matrix[:,39],np.log1p(5))
assert np.allclose(matrix[:,43],[.7,.3,.7]) and np.allclose(matrix[:,44],np.log1p(10))
q=SimpleNamespace(query_norm='unknown',search_location_id=999)
matrix=ns['v5_features'](q,(np.arange(3),None),history)
assert np.all(matrix[:,38]==0) and np.isnan(matrix[:,40:42]).all()
assert np.all(matrix[:,42:]==0)
print('Passed: multiple positives, missing-positive isolation, deterministic hard mining,')
print('38-feature compatibility, history availability, transition probabilities and missing geography.')
