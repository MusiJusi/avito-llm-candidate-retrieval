"""Independently validate a v8 CSV against the supplied raw benchmark files."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.inspection_deps')]
import pyarrow.parquet as pq

def validate():
    manifest=json.loads((ROOT/'artifacts/combined-pool-v8/manifest.json').read_text())
    queries=pq.read_table(ROOT/'benchmark_queries.parquet',columns=['query_id']).column(0).to_pylist()
    items=set(pq.read_table(ROOT/'benchmark_items.parquet',columns=['item_id']).column(0).to_pylist())
    path=ROOT/manifest['answer_file']
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==['query_id','answer']
        rows=list(reader)
    ids=[r['query_id'] for r in rows]
    assert len(rows)==len(queries)==2452 and len(set(ids))==len(ids) and set(ids)==set(queries)
    for row in rows:
        assert None not in row and len(row['query_id'])==16
        values=row['answer'].split(' ')
        assert row['answer']==' '.join(values) and len(values)==len(set(values))==50
        assert all(re.fullmatch('[0-9a-f]{16}',v) and v in items for v in values)
    with (ROOT/'answer.csv').open(encoding='utf-8',newline='') as stream:
        previous={r['query_id']:set(r['answer'].split(' ')) for r in csv.DictReader(stream)}
    changed=sum(previous[r['query_id']]!=set(r['answer'].split(' ')) for r in rows)
    report={'rows':len(rows),'candidates_per_query':50,'all_query_ids_exact':True,
        'all_item_ids_exact':True,'duplicates':0,'utf8':True,'columns':['query_id','answer'],
        'changed_top50_sets':changed,'answer_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert report['answer_sha256']==manifest['answer_sha256']
    (ROOT/'artifacts/combined-pool-v8/format_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':validate()
