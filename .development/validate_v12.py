"""Check the frozen v12 CSV against the exact benchmark identifiers."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.inspection_deps'))
import pandas as pd


def validate(path=None):
    manifest = json.loads((ROOT / 'artifacts/quality-v12/manifest.json').read_text(encoding='utf-8'))
    csv_path = ROOT / (path or manifest['answer_file'])
    raw = csv_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == manifest['answer_sha256'], (digest, manifest['answer_sha256'])
    assert raw.startswith(b'query_id,answer\n') and not raw.startswith(b'\xef\xbb\xbf')
    answer = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    queries = pd.read_parquet(ROOT / 'benchmark_queries.parquet', columns=['query_id'])
    items = pd.read_parquet(ROOT / 'benchmark_items.parquet', columns=['item_id'])
    expected = set(queries.query_id.astype(str))
    allowed = set(items.item_id.astype(str))
    assert list(answer.columns) == ['query_id', 'answer']
    assert len(answer) == len(queries) == 2452
    assert answer.query_id.is_unique and set(answer.query_id) == expected
    for row in answer.itertuples(index=False):
        assert isinstance(row.query_id, str) and len(row.query_id) == 16
        ids = row.answer.split(' ') if row.answer else []
        assert 0 < len(ids) <= 50 and len(ids) == len(set(ids))
        assert all(re.fullmatch('[0-9a-f]{16}', item) and item in allowed for item in ids)
    report = {'rows': len(answer), 'valid_query_ids': True, 'valid_item_ids': True,
              'answer_sha256': digest}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    validate()
