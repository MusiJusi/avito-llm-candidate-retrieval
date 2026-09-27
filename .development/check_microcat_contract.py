"""Check multiple positives, unsupported classes and the declared input boundary."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ['.inspection_deps','.semantic_deps','.development']]
import numpy as np
import pandas as pd
from microcat_models import micro_inputs, aggregated_targets, sparse_inputs, fit_nb, predict_micro, micro_candidate_features

history=pd.DataFrame({'query_norm':['ремонт','ремонт','ремонт','уборка'],
    'search_infm_params_text':['Тип услуги Ремонт','','',''],
    'item_microcat_id':[10,10,20,30], 'search_category':[1,2,3,4],
    'search_location_id':[1,1,2,3], 'item_id':['a','b','c','d']})
inputs=micro_inputs(history).drop_duplicates('key').reset_index(drop=True)
space={'input_to_row':dict(zip(inputs.key,range(len(inputs)))), 'classes':np.array([10,20,30,40]),
       'sparse':sparse_inputs(inputs)}
rows,targets,totals,supported=aggregated_targets(history,space['classes'],space['input_to_row'])
position=space['input_to_row'][micro_inputs(history).key.iloc[1]]
assert set(space['classes'][targets[np.flatnonzero(rows==position)[0]]>0])=={10,20}
changed=history.assign(search_category=999,search_location_id=999,item_id='other')
assert micro_inputs(changed).key.equals(micro_inputs(history).key)
model=fit_nb(space,history)
probabilities=predict_micro(model,space,history)
assert np.allclose(probabilities.sum(axis=1),1) and np.all(probabilities[:,3]==0)
features=micro_candidate_features(probabilities[0],np.array([10,40,999]),space['classes'],model['supported'])
assert features[0,-1]==1 and np.all(features[1:,-1]==0)
assert np.isnan(features[1:,0]).all(), 'Unseen microcats must not become hard-negative scores'
print('Passed: multi-positive targets, category/location/ID input exclusion, normalized probabilities and unsupported microcats.')
