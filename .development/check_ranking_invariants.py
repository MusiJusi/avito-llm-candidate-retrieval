"""Exercise text-group OOF, missing positives, context sampling and RRF contracts."""
import ast,json,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.inspection_deps'))
import numpy as np
import pandas as pd
import joblib
import gc
from collections import Counter
from scipy.stats import rankdata
nb=json.loads((root/'Avito.ipynb').read_text(encoding='utf-8'))
functions={}
for cell in nb['cells']:
    if cell['cell_type']=='code':
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef):functions[node.name]=node
workspace=root/'.verification'
workspace.mkdir(exist_ok=True)
cache=Path(tempfile.mkdtemp(prefix='ranking_invariants_',dir=workspace))
assert cache.resolve().is_relative_to(root.resolve())
ids=np.array([f'{i:016x}' for i in range(60)])
ns={'np':np,'pd':pd,'joblib':joblib,'gc':gc,'json':json,'Counter':Counter,'rankdata':rankdata,
    'ITEM_IDS':ids,'ITEM_TO_ROW':{x:i for i,x in enumerate(ids)},'SEED':260926,
    'CONFIG':{'cold_item_fraction':.9},'EXPERIMENT_CONFIG':{'contexts_per_text':4,
        'hard_source':2,'hard_model':2,'random_tail':2},'CACHE':cache,'USE_CACHE':False,
    'fingerprint':'fixture','prior_legacy':None,
    'QUERY_COLS':['search_query','search_location_id','search_infm_params_text','search_is_delivery_search','search_category']}
for name in ['stable_topk','save_cache','query_labels','purge_history','context_sample',
             'grouped_training','direct_predict','combined_predict','seen_query_history','strata','matched_weights']:
    exec(compile(ast.Module(body=[functions[name]],type_ignores=[]),'<actual notebook function>','exec'),ns)
frame=pd.DataFrame({'search_query':['alpha','alpha','beta','gamma'],
    'query_norm':['alpha','alpha','beta','gamma'],'context_key':['a1','a2','b1','g1'],
    'search_location_id':[1,2,1,1],'search_infm_params_text':['','','',''],
    'search_is_delivery_search':[0]*4,'search_category':[114]*4})
rows=[]
for position,truth in enumerate([[0,1],[2],[4],[3]]):
    for item in truth:
        row=frame.iloc[position].to_dict();row['item_id']=ids[item];rows.append(row)
history=pd.DataFrame(rows)
calls=[]
class FixtureHistory:
    def __init__(self,fit):self.fit=fit
def retrieve(q,h,progress_every):
    assert not set(q.query_norm)&set(h.fit.query_norm)
    calls.append(tuple(q.context_key))
    # beta's positive ID 4 is intentionally absent. It must not be injected.
    record=(np.arange(4),np.tile(np.arange(4)[:,None],(1,38)).astype(np.float32))
    return [record for _ in range(len(q))]
ns.update({'HistorySignals':FixtureHistory,'retrieve_semantic_features':retrieve,
    'rank_semantic_features':lambda q,r:r[1],
    'model_scores':lambda m,f:[x[:,0] for x in f],
    'score_candidates':lambda x,c:x[:,0],'best_config':{}})
data=ns['grouped_training'](history,frame,'fixture_training')
assert set(data['group'])=={0,1,3}, 'Missing positive was injected or a valid group lost'
assert all(row['text_overlap']==row['purged_item_overlap']==0 for row in data['audit'])
assert any(set(c)=={'a1','a2'} for c in calls), 'Contexts of one text were split across OOF folds'
assert not any(('a1' in c)^('a2' in c) for c in calls)
assert int(data['y'][data['group']==0].sum())==2, 'Multiple positives lost'
again=ns['grouped_training'](history,frame,'fixture_repeat')
for key in ['X','y','weight','group']:assert np.array_equal(data[key],again[key])

records=[(np.arange(60),np.zeros((60,10),dtype=np.float32))]
prior=[np.arange(60,dtype=float)];learned=[-prior[0]]
assert np.array_equal(ns['combined_predict'](records,learned,prior,0)[0],ns['direct_predict'](records,prior)[0])
assert np.array_equal(ns['combined_predict'](records,learned,prior,1)[0],ns['direct_predict'](records,learned)[0])
ns.update({'history_all':history,'new_audit':pd.DataFrame({'query_norm':['unused held text']}),'new_cold_ids':set()})
seen=ns['seen_query_history'](frame.iloc[:1],[{0,1}])
assert 'a1' not in set(seen.context_key)
assert 'a2' in set(seen.context_key), 'Other contexts of the known text were removed'
ns['centers']=pd.DataFrame(index=[1,2])
stratum_frame=frame.iloc[:2].copy()
stratum_frame['query_norm']=['short','a considerably longer query text']
stratum_frame['search_infm_params_text']=['','rating 4']
target=pd.concat([stratum_frame.iloc[:1]]*3+[stratum_frame.iloc[1:]],ignore_index=True)
assert np.allclose(ns['matched_weights'](stratum_frame,target),[.75,.25])
missing=target.copy();missing['search_location_id']=999
try:ns['matched_weights'](stratum_frame,missing)
except AssertionError:pass
else:raise AssertionError('Unsupported benchmark stratum was silently ignored')
sample_history=pd.concat([frame.iloc[:1]]*10,ignore_index=True)
sample_history['context_key']=[f'key{i}' for i in range(10)]
sample_history['item_id']=ids[:10]
sample=ns['context_sample'](sample_history,100,123,sample_history.iloc[:1])
assert 'key0' in set(sample.context_key)
assert sample.context_key.is_unique and len(sample)<=5
print('Passed: text-group OOF, own-label purge, no positive injection, multiple positives, determinism,')
print('RRF endpoints, held context isolation, known-text history, distribution weights, context cap')
