"""Fresh CPU-only, network-disabled reproduction of the portable v11 answer."""
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


def verify():
    manifest = json.loads((ROOT/'artifacts/bge-m3-v11/manifest.json').read_text(encoding='utf-8'))
    reference = (ROOT/manifest['answer_file']).read_bytes()
    assert hashlib.sha256(reference).hexdigest() == manifest['answer_sha256']
    directory = Path(tempfile.mkdtemp(prefix='quality_v11_cpu_',
                                      dir=ROOT/'.verification'))
    assert directory.resolve().is_relative_to(ROOT/'.verification')
    with zipfile.ZipFile(ROOT/'deliverables/avito_v11_solution.zip') as archive:
        for name in archive.namelist():
            assert (directory/name).resolve().is_relative_to(directory)
        archive.extractall(directory)
    output = directory/'answer.csv'
    output.unlink()
    for name in ['train.parquet', 'benchmark_items.parquet',
                 'benchmark_queries.parquet']:
        os.link(ROOT/name, directory/name)
    environment = os.environ.copy()
    environment.update(AVITO_DISABLE_NETWORK='1', AVITO_REBUILD_CACHE='0',
                       AVITO_RUN_MODEL_SEARCH='0', PYTHONIOENCODING='utf-8',
                       CUDA_VISIBLE_DEVICES='-1')
    environment['PYTHONPATH'] = os.pathsep.join(str(ROOT/folder) for folder in
        ['.ranking_deps', '.inspection_deps', '.semantic_deps'])
    started = time.perf_counter()
    log = ROOT/'artifacts/bge-m3-v11/offline_verification.log'
    print('Verify portable v11', directory, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-u',
            str(directory/'.development/run_validation_notebook.py'),
            'Avito.ipynb'], cwd=directory, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    execution = json.loads((directory/'artifacts/notebook_execution.json').read_text())
    assert execution['device'] == 'cpu' and execution['network_connections_disabled']
    assert output.read_bytes() == reference
    report = {'fresh_extraction': True, 'network_connections_disabled': True,
              'answer_bytes_equal': True, 'answer_sha256': manifest['answer_sha256'],
              'notebook_code_sha256': execution['code_sha256'],
              'code_cells_executed': execution['code_cells_executed'],
              'elapsed_seconds': round(time.perf_counter()-started, 2),
              'verification_directory': directory.relative_to(ROOT).as_posix()}
    (ROOT/'artifacts/bge-m3-v11/reproduction.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    (ROOT/'deliverables/v11/Avito.ipynb').write_bytes((directory/'Avito.ipynb').read_bytes())
    print('v11 reproduced', json.dumps(report), flush=True)


if __name__ == '__main__':
    verify()
