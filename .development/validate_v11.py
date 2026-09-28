"""Independent format and identifier validation for the selected v11 CSV."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.inspection_deps'))
import pandas as pd


def validate():
    manifest = json.loads((ROOT/'artifacts/bge-m3-v11/manifest.json').read_text(encoding='utf-8'))
    path = ROOT/manifest['answer_file']
    data = path.read_bytes()
    assert not data.startswith(b'\xef\xbb\xbf'), 'UTF-8 BOM is not expected'
    assert data.split(b'\n', 1)[0] == b'query_id,answer'
    assert hashlib.sha256(data).hexdigest() == manifest['answer_sha256']
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    queries = pd.read_parquet(ROOT/'benchmark_queries.parquet', columns=['query_id'])
    items = pd.read_parquet(ROOT/'benchmark_items.parquet', columns=['item_id'])
    assert list(answer.columns) == ['query_id', 'answer']
    assert len(answer) == len(queries) == 2452
    assert answer.query_id.is_unique
    expected = set(queries.query_id.astype(str))
    allowed = set(items.item_id.astype(str))
    assert set(answer.query_id) == expected
    for row in answer.itertuples(index=False):
        assert isinstance(row.query_id, str) and len(row.query_id) == 16
        ids = row.answer.split(' ') if row.answer else []
        assert 0 < len(ids) <= 50 and len(ids) == len(set(ids))
        assert all(re.fullmatch(r'[0-9a-f]{16}', item) and item in allowed
                   for item in ids)
    print(json.dumps({'rows': len(answer), 'all_ids_valid': True,
                      'answer_sha256': manifest['answer_sha256']}, indent=2))


if __name__ == '__main__':
    validate()
