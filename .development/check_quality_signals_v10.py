"""Small meaningful contracts for missing category and held history exclusion."""
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.inspection_deps'),str(ROOT/'.ranking_deps')]
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from quality_signals_v10 import QualityEvidence,query_filter_text

history=pd.DataFrame({'search_location_id':[1,1], 'item_id':['a','b'],
    'query_norm':['alpha','beta'], 'item_microcat_id':[10,20], 'item_location_id':[1,2]})
ids=np.array(['a','held']);categories=np.array([114,114]);microcats=np.array([10,20]);locations=np.array([1,2])
vectors=np.array([[1.,0.],[0.,1.]],np.float32)
query=SimpleNamespace(search_location_id=1,search_category=0,search_is_delivery_search=0)
bank=QualityEvidence(history)
x=bank.features(query,np.arange(2),ids,categories,microcats,locations,vectors,
    np.array([1.,0.]),np.zeros(2),rankdata)
assert x.shape==(2,12) and not np.isinf(x).any()
assert not x[:,0].any(),'Category 0 must not act as a category requirement'
assert x[1,5]==x[1,6]==x[1,7]==0,'Held item must not acquire historical popularity'
query.search_category=114
assert bank.features(query,np.arange(2),ids,categories,microcats,locations,vectors,
    np.array([1.,0.]),np.zeros(2),rankdata)[:,0].all()
empty=QualityEvidence(history.iloc[:0])
assert empty.features(query,np.arange(2),ids,categories,microcats,locations,vectors,
    np.array([1.,0.]),np.zeros(2),rankdata).shape==(2,12)
assert query_filter_text('alpha','  x\n y ')==query_filter_text('alpha','x y')
print('Quality signal contracts passed.')
