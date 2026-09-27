"""Validate the actual CSV against the original data and compare with v0.3."""
import csv,hashlib,io,json,re,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.inspection_deps'))
import pandas as pd
queries=pd.read_parquet(root/'benchmark_queries.parquet',columns=['query_id'])
items=pd.read_parquet(root/'benchmark_items.parquet',columns=['item_id'])
raw=(root/'answer.csv').read_bytes()
rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
assert list(rows[0])==['query_id','answer']
qids=[row['query_id'] for row in rows]
assert len(rows)==len(queries)==len(set(qids))
assert set(qids)==set(queries.query_id)
allowed=set(items.item_id)
sizes=[]
for row in rows:
    assert isinstance(row['query_id'],str) and len(row['query_id'])==16
    ids=row['answer'].split(' ')
    assert 0<len(ids)<=50 and len(ids)==len(set(ids))
    assert all(re.fullmatch('[0-9a-f]{16}',item) and item in allowed for item in ids)
    sizes.append(len(ids))
old=subprocess.check_output(['git','show','v0.3.0:answer.csv'],cwd=root).decode('utf-8')
previous={row['query_id']:set(row['answer'].split(' ')) for row in csv.DictReader(io.StringIO(old))}
changed=sum(set(row['answer'].split(' '))!=previous[row['query_id']] for row in rows)
overlap=sum(len(set(row['answer'].split(' '))&previous[row['query_id']]) for row in rows)/len(rows)
report={'valid':True,'columns':['query_id','answer'],'queries':len(rows),
        'min_candidates':min(sizes),'max_candidates':max(sizes),
        'identifier_case_and_corpus_membership_valid':True,'utf8_valid':True,
        'single_space_separator':True,'answer_sha256':hashlib.sha256(raw).hexdigest(),
        'changed_sets_vs_v03':changed,'mean_overlap_items_vs_v03':overlap}
(root/'artifacts/ranking-v1/submission_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
