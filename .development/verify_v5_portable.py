"""Run the extracted v5 deliverable on CPU with network connections blocked."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/ranking-v5'
reference=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
workspace=ROOT/'.verification'
workspace.mkdir(exist_ok=True)
destination=Path(tempfile.mkdtemp(prefix='v5_portable_cpu_',dir=workspace))
assert destination.resolve().is_relative_to(ROOT.resolve())
with zipfile.ZipFile(ROOT/'deliverables/avito_v5_solution.zip') as archive:
    for name in archive.namelist():
        assert (destination/name).resolve().is_relative_to(destination.resolve())
    archive.extractall(destination)
for name in ['train.parquet','benchmark_items.parquet','benchmark_queries.parquet']:
    os.link(ROOT/name,destination/name)
environment=os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='-1',AVITO_DISABLE_NETWORK='1',
                   AVITO_REBUILD_CACHE='0',AVITO_RUN_MODEL_SEARCH='0',PYTHONIOENCODING='utf-8')
environment['PYTHONPATH']=os.pathsep.join(str(ROOT/name) for name in
    ['.ranking_deps','.semantic_deps','.inspection_deps'])
started=time.perf_counter()
subprocess.run([sys.executable,'-u',str(destination/'.development/run_validation_notebook.py'),'Avito.ipynb'],
               cwd=destination,env=environment,check=True)
execution=json.loads((destination/'artifacts/notebook_execution.json').read_text())
result=json.loads((destination/'artifacts/ranking-v5/manifest.json').read_text())
digest=hashlib.sha256((destination/'answer.csv').read_bytes()).hexdigest()
assert execution['device']=='cpu' and execution['network_connections_disabled']
assert digest==reference['answer_sha256']
assert result['selection']==reference['selection'] and result['control']==reference['control']
assert result['pipeline_sha256']==reference['pipeline_sha256']
report={'fresh_extracted_directory':True,'original_search_caches_available':False,
        'device':'cpu','network_connections_disabled':True,'answer_bytes_equal':True,
        'frozen_selection_and_control_metadata_equal':True,'answer_sha256':digest,
        'notebook_code_sha256':execution['code_sha256'],
        'elapsed_seconds':round(time.perf_counter()-started,2),
        'code_cells_executed':execution['code_cells_executed']}
(CACHE/'portable_reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
print('Verification directory:',destination,flush=True)
