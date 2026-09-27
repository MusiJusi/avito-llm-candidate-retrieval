"""Prove that the normalized primary notebook recreates the submitted CSV offline."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
reference = json.loads((ROOT/'artifacts/current_solution.json').read_text())
parent = ROOT/'.verification'
parent.mkdir(exist_ok=True)
destination = Path(tempfile.mkdtemp(prefix='current_cpu_', dir=parent))
assert destination.resolve().is_relative_to(ROOT)
with zipfile.ZipFile(ROOT/'deliverables/avito_solution.zip') as archive:
    for name in archive.namelist():
        assert (destination/name).resolve().is_relative_to(destination)
    archive.extractall(destination)
csv = destination/'answer.csv'
csv.unlink()
for name in ['train.parquet', 'benchmark_items.parquet', 'benchmark_queries.parquet']:
    os.link(ROOT/name, destination/name)
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='-1', AVITO_DISABLE_NETWORK='1',
    AVITO_REBUILD_CACHE='0', AVITO_RUN_MODEL_SEARCH='0', PYTHONIOENCODING='utf-8')
environment['PYTHONPATH'] = os.pathsep.join(str(ROOT/name) for name in
    ['.ranking_deps', '.semantic_deps', '.inspection_deps'])
started = time.perf_counter()
subprocess.run([sys.executable, '-u', str(destination/'.development/run_validation_notebook.py'),
    'Avito.ipynb'], cwd=destination, env=environment, check=True)
execution = json.loads((destination/'artifacts/notebook_execution.json').read_text())
assert execution['device'] == 'cpu' and execution['network_connections_disabled']
assert csv.read_bytes() == (ROOT/'answer.csv').read_bytes()
assert hashlib.sha256(csv.read_bytes()).hexdigest() == reference['answer_sha256']
report = {'fresh_extraction': True, 'existing_answer_removed': True,
    'original_pool_caches_available': False, 'device': 'cpu',
    'network_connections_disabled': True, 'answer_bytes_equal': True,
    'answer_sha256': reference['answer_sha256'], 'notebook_code_sha256': execution['code_sha256'],
    'code_cells_executed': execution['code_cells_executed'],
    'elapsed_seconds': round(time.perf_counter()-started, 2)}
(ROOT/'artifacts/current_reproduction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
# Retain real outputs in the primary notebook without changing its source code.
(ROOT/'Avito.ipynb').write_bytes((destination/'Avito.ipynb').read_bytes())
print(json.dumps(report, indent=2), flush=True)
