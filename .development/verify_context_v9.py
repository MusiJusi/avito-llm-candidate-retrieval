"""Fresh CPU reproduction, with network disabled and source CSV removed."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def verify():
    manifest=json.loads((ROOT/'artifacts/context-v9/manifest.json').read_text())
    reference=(ROOT/manifest['answer_file']).read_bytes()
    parent=ROOT/'.verification';parent.mkdir(exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='context_cpu_',dir=parent))
    assert destination.resolve().is_relative_to(ROOT)
    with zipfile.ZipFile(ROOT/'deliverables/avito_v9_solution.zip') as archive:
        for name in archive.namelist():assert (destination/name).resolve().is_relative_to(destination)
        archive.extractall(destination)
    csv=destination/'answer.csv';csv.unlink()
    for name in ['train.parquet','benchmark_items.parquet','benchmark_queries.parquet']:
        os.link(ROOT/name,destination/name)
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='-1',AVITO_DISABLE_NETWORK='1',
        AVITO_REBUILD_CACHE='0',AVITO_RUN_MODEL_SEARCH='0',PYTHONIOENCODING='utf-8')
    environment['PYTHONPATH']=os.pathsep.join(str(ROOT/name) for name in
        ['.ranking_deps','.semantic_deps','.inspection_deps'])
    started=time.perf_counter()
    log=ROOT/'artifacts/context-v9/offline_verification.log'
    print('Fresh CPU verification',destination.relative_to(ROOT).as_posix(),flush=True)
    with log.open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',str(destination/'.development/run_validation_notebook.py'),
            'Avito.ipynb'],cwd=destination,env=environment,check=True,stdout=stream,stderr=subprocess.STDOUT)
    execution=json.loads((destination/'artifacts/notebook_execution.json').read_text())
    assert execution['device']=='cpu' and execution['network_connections_disabled']
    assert csv.read_bytes()==reference
    report={'fresh_extraction':True,'existing_answer_removed':True,'original_pool_caches_available':False,
        'device':'cpu','network_connections_disabled':True,'answer_bytes_equal':True,
        'answer_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
        'notebook_code_sha256':execution['code_sha256'],'code_cells_executed':execution['code_cells_executed'],
        'elapsed_seconds':round(time.perf_counter()-started,2),
        'verification_directory':destination.relative_to(ROOT).as_posix()}
    assert report['answer_sha256']==manifest['answer_sha256']
    (ROOT/'artifacts/context-v9/reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'deliverables/v9/Avito.ipynb').write_bytes((destination/'Avito.ipynb').read_bytes())
    assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()=='a256639b72ef3bd515bf5118c5e337aee23a34c9e0b261da580b1c852411a6b2'
    print('Context v9 reproduction',json.dumps(report),flush=True)
if __name__=='__main__':verify()
