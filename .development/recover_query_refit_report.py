"""Recover report-only I/O failure after all three training epochs completed.

The pilot helper expects a specific document-vector filename. Full refit uses
another content-addressed filename. Link the verified vector cache to that alias;
do not change weights, query vectors, fingerprints or training hyperparameters.
Epoch loss means were in process memory and are deliberately not reconstructed.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.inspection_deps'))
import numpy as np
import pandas as pd
CACHE=ROOT/'artifacts/query-encoder-candidate'
qe=json.loads((ROOT/'artifacts/query-encoder-pilot/training_report.json').read_text())['fingerprint']
documents=list(CACHE.glob('documents_*.npy'))
assert len(documents)==1,'Ambiguous refit cache: inspect fingerprints before recovery'
ids=pd.read_parquet(ROOT/'train.parquet',columns=['item_id']).item_id.nunique()
assert np.load(documents[0],mmap_mode='r').shape==(ids,384)
vectors=CACHE/f'queries_epoch_3_{qe}.npy'
queries=pd.read_parquet(ROOT/'benchmark_queries.parquet',columns=['search_query'])
assert vectors.exists() and np.isfinite(np.load(vectors,allow_pickle=False)).all()
model=CACHE/f'epoch_3_{qe}/model.safetensors'
assert model.exists() and model.stat().st_size>400*2**20
alias=CACHE/f'document_vectors_{qe}.npy'
if not alias.exists():os.link(documents[0],alias)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
report={'fingerprint':qe,'full_train_refit':True,'training_completed':True,
    'input_sha256':json.loads((ROOT/'artifacts/ranking-v5/manifest.json').read_text())['input_sha256'],
    'config':json.loads((ROOT/'artifacts/query-encoder-pilot/training_report.json').read_text())['config'],
    'document_vectors':ids,'document_vectors_sha256':sha(documents[0]),
    'encoder_sha256':sha(model),'query_vectors_sha256':sha(vectors),
    'epoch_checkpoints':[1,3],'logs':None,'report_recovered':True,
    'reason':'Report-only missing alias after epoch 3 weights and benchmark query vectors were saved. Loss means unavailable; no weights changed.'}
(CACHE/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Recovered completed full-train model metadata; weights and vectors unchanged.')
