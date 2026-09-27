"""Verify the separate candidate notebook on CPU, offline, in a fresh extraction."""
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
CACHE=ROOT/'artifacts/microcat-v6'
reference=json.loads((CACHE/'manifest.json').read_text())
parent=ROOT/'.verification'
parent.mkdir(exist_ok=True)
destination=Path(tempfile.mkdtemp(prefix='microcat_cpu_',dir=parent))
assert destination.resolve().is_relative_to(ROOT.resolve())
with zipfile.ZipFile(ROOT/'deliverables/avito_v6_experiments.zip') as archive:
    for name in archive.namelist():assert (destination/name).resolve().is_relative_to(destination.resolve())
    archive.extractall(destination)
for name in ['train.parquet','benchmark_items.parquet','benchmark_queries.parquet']:
    os.link(ROOT/name,destination/name)
environment=os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='-1',AVITO_DISABLE_NETWORK='1',AVITO_REBUILD_CACHE='0',
                   AVITO_RUN_MODEL_SEARCH='0',PYTHONIOENCODING='utf-8')
environment['PYTHONPATH']=os.pathsep.join(str(ROOT/name) for name in ['.ranking_deps','.semantic_deps','.inspection_deps'])
started=time.perf_counter()
subprocess.run([sys.executable,'-u',str(destination/'.development/run_validation_notebook.py'),
               'Avito_microcat_candidate_v6.ipynb'],cwd=destination,env=environment,check=True)
result=json.loads((destination/'artifacts/microcat-v6/manifest.json').read_text())
execution=json.loads((destination/'artifacts/notebook_execution.json').read_text())
assert execution['device']=='cpu' and execution['network_connections_disabled']
assert hashlib.sha256((destination/reference['answer_file']).read_bytes()).hexdigest()==reference['answer_sha256']
assert result['selection']==reference['selection'] and result['control']==reference['control']
assert (destination/'answer.csv').read_bytes()==(ROOT/'answer.csv').read_bytes()
report={'fresh_extracted_directory':True,'original_pool_caches_available':False,'device':'cpu',
    'network_connections_disabled':True,'candidate_bytes_equal':True,'main_v5_answer_unchanged':True,
    'answer_sha256':reference['answer_sha256'],'notebook_code_sha256':execution['code_sha256'],
    'code_cells_executed':execution['code_cells_executed'],'elapsed_seconds':round(time.perf_counter()-started,2)}
(CACHE/'portable_reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
