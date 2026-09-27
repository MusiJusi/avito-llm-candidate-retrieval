"""Validate the current answer against raw IDs and the confirmed submission hash."""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.inspection_deps'))
import pandas as pd

def validate():
    current = json.loads((ROOT/'artifacts/current_solution.json').read_text())
    raw = (ROOT/current['answer_file']).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
    assert reader.fieldnames == ['query_id', 'answer']
    rows = list(reader)
    queries = pd.read_parquet(ROOT/'benchmark_queries.parquet', columns=['query_id'])
    items = pd.read_parquet(ROOT/'benchmark_items.parquet', columns=['item_id'])
    identifiers = [r['query_id'] for r in rows]
    assert len(identifiers) == len(queries) == 2452
    assert len(set(identifiers)) == len(identifiers) and set(identifiers) == set(queries.query_id)
    allowed = set(items.item_id)
    for row in rows:
        assert len(row['query_id']) == 16
        ids = row['answer'].split(' ')
        assert len(ids) == 50 and len(set(ids)) == 50
        assert all(re.fullmatch('[0-9a-f]{16}', item) and item in allowed for item in ids)
    assert hashlib.sha256(raw).hexdigest() == current['answer_sha256']
    for name, digest in current['immutable_training_source_sha256'].items():
        assert hashlib.sha256((ROOT/'.development'/name).read_bytes()).hexdigest() == digest
    report = {'valid': True, 'file': current['answer_file'], 'queries': len(rows),
        'candidates_per_query': 50, 'platform_recall50': current['platform_recall50'],
        'answer_sha256': current['answer_sha256'], 'numerical_training_sources_unchanged': True}
    (ROOT/'artifacts/current_submission_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
