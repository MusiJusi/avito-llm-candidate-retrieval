"""Validate the experimental CSV independently while protecting the main answer."""
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.inspection_deps'))
import pandas as pd
CACHE=ROOT/'artifacts/microcat-v6'
manifest=json.loads((CACHE/'manifest.json').read_text())
raw=(ROOT/manifest['answer_file']).read_bytes()
reader=csv.DictReader(io.StringIO(raw.decode('utf-8')))
assert reader.fieldnames==['query_id','answer']
rows=list(reader)
queries=pd.read_parquet(ROOT/'benchmark_queries.parquet',columns=['query_id'])
items=pd.read_parquet(ROOT/'benchmark_items.parquet',columns=['item_id'])
ids=[row['query_id'] for row in rows]
assert len(rows)==len(queries)==len(set(ids)) and set(ids)==set(queries.query_id)
allowed=set(items.item_id)
for row in rows:
    assert len(row['query_id'])==16
    values=row['answer'].split(' ')
    assert len(values)==50 and len(set(values))==50
    assert all(re.fullmatch('[0-9a-f]{16}',value) and value in allowed for value in values)
digest=hashlib.sha256(raw).hexdigest()
assert digest==manifest['answer_sha256']
previous=json.loads((ROOT/'artifacts/ranking-v5/manifest.json').read_text())
assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()==previous['answer_sha256']
report={'valid':True,'file':manifest['answer_file'],'queries':len(rows),'candidates_per_query':50,
        'ids_utf8_header_and_spacing_valid':True,'answer_sha256':digest,'main_v5_answer_unchanged':True}
(CACHE/'submission_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
