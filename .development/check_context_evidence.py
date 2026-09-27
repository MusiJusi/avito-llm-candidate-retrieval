"""Check information boundaries on tiny data, without fitting the real pipeline."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.inspection_deps'),str(ROOT/'.development')]
import numpy as np
import pandas as pd
from context_signals_v9 import GeographicEvidence,SemanticEvidence

history=pd.DataFrame({'query_norm':['paint','paint','wall paint'],
    'item_id':['a','b','a'],'item_microcat_id':[1,1,1],
    'search_location_id':[10,10,10],'item_location_id':[20,20,20]})
coordinates=pd.DataFrame({'lat':[55.75,55.76,-40.], 'lon':[37.6,37.61,120.]},index=['a','b','held'])
first=GeographicEvidence(history,coordinates,{})
modified=coordinates.copy();modified.loc['held']=[80.,-120.]
second=GeographicEvidence(history,modified,{})
assert np.array_equal(first.evidence(10)[0],second.evidence(10)[0])
assert first.evidence(10)[1][0]==1 and first.evidence(999)[1][0]==2
assert first.evidence(10)[1][4]<.2,'A tiny history cannot become a confident city center'
index=SemanticEvidence(history,{'paint':np.array([1.,0.,0.]),'wall paint':np.array([.8,.6,0.])},
    ['a','b','held'],np.eye(3,dtype=np.float32),[1,2])
prototype,probability,stats=index.query_evidence(np.array([[1.,0.,0.]],np.float32))
assert prototype[0,2]==0,'A held item must not contaminate positive centroids'
assert probability[0,1]==0,'A held microcategory must not contaminate the history'
assert np.isfinite(stats).all() and abs(np.linalg.norm(prototype[0])-1)<1e-6
print('History boundaries passed: held metadata/labels excluded, unknown center explicit, sparse geography uncertain.')
