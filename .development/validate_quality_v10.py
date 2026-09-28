"""Strict CSV contract and comparison with the confirmed v9 submission."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.inspection_deps')]
import pandas as pd


def validate():
    manifest=json.loads((ROOT/'artifacts/quality-v10/manifest.json').read_text())
    path=ROOT/manifest['answer_file'];data=path.read_bytes()
    assert data.decode('utf-8').splitlines()[0]=='query_id,answer'
    queries=pd.read_parquet(ROOT/'benchmark_queries.parquet',columns=['query_id']).query_id.astype(str)
    items=set(pd.read_parquet(ROOT/'benchmark_items.parquet',columns=['item_id']).item_id.astype(str))
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);assert reader.fieldnames==['query_id','answer'];rows=list(reader)
    assert len(rows)==len(queries)==2452
    observed=[row['query_id'] for row in rows]
    assert len(set(observed))==len(observed) and set(observed)==set(queries)
    counts=[]
    for row in rows:
        assert len(row['query_id'])==16
        ids=row['answer'].split(' ')
        assert len(ids)==len(set(ids))==50
        assert all(re.fullmatch('[0-9a-f]{16}',item) and item in items for item in ids)
        counts.append(len(ids))
    old=pd.read_csv(ROOT/'experiments/results/v9/answer.csv',dtype=str,keep_default_na=False).set_index('query_id').answer
    changed=sum(set(row['answer'].split(' '))!=set(old[row['query_id']].split(' ')) for row in rows)
    report={'valid':True,'rows':len(rows),'columns':['query_id','answer'],'encoding':'utf-8',
        'minimum_candidates':min(counts),'maximum_candidates':max(counts),'unknown_items':0,'duplicate_items':0,
        'answer_sha256':hashlib.sha256(data).hexdigest(),'changed_sets_vs_v9':changed}
    assert report['answer_sha256']==manifest['answer_sha256']
    (ROOT/'artifacts/quality-v10/answer_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':validate()
