"""Independent CSV contract validation for the learned-query candidate."""
from pathlib import Path
import csv
import io
import json
import hashlib
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.inspection_deps'))
import pandas as pd
CACHE=ROOT/'artifacts/query-encoder-candidate'
manifest=json.loads((CACHE/'manifest.json').read_text())
raw=(ROOT/manifest['answer_file']).read_bytes()
reader=csv.DictReader(io.StringIO(raw.decode('utf-8')))
assert reader.fieldnames==['query_id','answer']
rows=list(reader)
queries=pd.read_parquet(ROOT/'benchmark_queries.parquet',columns=['query_id'])
items=pd.read_parquet(ROOT/'benchmark_items.parquet',columns=['item_id'])
query_ids=[row['query_id'] for row in rows]
assert len(query_ids)==len(queries)==len(set(query_ids)) and set(query_ids)==set(queries.query_id)
allowed=set(items.item_id)
for row in rows:
    assert len(row['query_id'])==16
    ids=row['answer'].split(' ')
    assert len(ids)==50 and len(set(ids))==50
    assert all(re.fullmatch('[0-9a-f]{16}',item) and item in allowed for item in ids)
digest=hashlib.sha256(raw).hexdigest()
assert digest==manifest['answer_sha256']
v5=json.loads((ROOT/'artifacts/ranking-v5/manifest.json').read_text())
assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()==v5['answer_sha256']
report={'valid':True,'file':manifest['answer_file'],'queries':len(rows),'candidates_per_query':50,
    'ids_utf8_header_and_spacing_valid':True,'answer_sha256':digest,'main_v5_answer_unchanged':True}
(CACHE/'submission_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
