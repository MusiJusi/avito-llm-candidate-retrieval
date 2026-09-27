"""Validate submission bytes against original IDs; compare to the preserved v4."""
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.inspection_deps'))
import pandas as pd

raw = (ROOT / 'answer.csv').read_bytes()
reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
assert reader.fieldnames == ['query_id', 'answer']
rows = list(reader)
queries = pd.read_parquet(ROOT / 'benchmark_queries.parquet', columns=['query_id'])
items = pd.read_parquet(ROOT / 'benchmark_items.parquet', columns=['item_id'])
query_ids = [row['query_id'] for row in rows]
assert len(rows) == len(queries) == len(set(query_ids))
assert set(query_ids) == set(queries.query_id)
allowed = set(items.item_id)
sizes = []
for row in rows:
    assert len(row['query_id']) == 16
    ids = row['answer'].split(' ')
    assert 0 < len(ids) <= 50 and len(ids) == len(set(ids))
    assert all(re.fullmatch('[0-9a-f]{16}', item) and item in allowed for item in ids)
    sizes.append(len(ids))
previous = {row['query_id']: set(row['answer'].split(' ')) for row in
            csv.DictReader(io.StringIO((ROOT / 'answer_v0.4.csv').read_text(encoding='utf-8')))}
report = {'valid': True, 'queries': len(rows), 'columns': reader.fieldnames,
          'min_candidates': min(sizes), 'max_candidates': max(sizes),
          'utf8_and_single_space_valid': True, 'all_ids_match_original_corpus': True,
          'answer_sha256': hashlib.sha256(raw).hexdigest(),
          'changed_sets_vs_v04': sum(set(row['answer'].split(' ')) != previous[row['query_id']] for row in rows),
          'mean_overlap_items_vs_v04': sum(len(set(row['answer'].split(' ')) & previous[row['query_id']]) for row in rows) / len(rows)}
manifest = json.loads((ROOT / 'artifacts/ranking-v5/manifest.json').read_text(encoding='utf-8'))
assert report['answer_sha256'] == manifest['answer_sha256']
(ROOT / 'artifacts/ranking-v5/submission_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
