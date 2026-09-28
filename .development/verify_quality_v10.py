"""Fresh offline execution; CPU by default, GPU for a selected cross-encoder."""
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
    manifest=json.loads((ROOT/'artifacts/quality-v10/manifest.json').read_text())
    reference=(ROOT/manifest['answer_file']).read_bytes()
    use_gpu=bool(manifest.get('cross_weight'))
    directory=Path(tempfile.mkdtemp(prefix='quality_v10_gpu_' if use_gpu else 'quality_v10_cpu_',dir=ROOT/'.verification'))
    assert directory.resolve().is_relative_to(ROOT/'.verification')
    with zipfile.ZipFile(ROOT/'deliverables/avito_v10_solution.zip') as archive:
        for name in archive.namelist():assert (directory/name).resolve().is_relative_to(directory)
        archive.extractall(directory)
    output=directory/'answer.csv'
    output.unlink()
    for name in ['train.parquet','benchmark_items.parquet','benchmark_queries.parquet']:
        os.link(ROOT/name,directory/name)
    environment=os.environ.copy()
    environment.update(AVITO_DISABLE_NETWORK='1',AVITO_REBUILD_CACHE='0',
        AVITO_RUN_MODEL_SEARCH='0',PYTHONIOENCODING='utf-8')
    if not use_gpu:environment['CUDA_VISIBLE_DEVICES']='-1'
    environment['PYTHONPATH']=os.pathsep.join(str(ROOT/folder) for folder in ['.ranking_deps','.inspection_deps','.semantic_deps'])
    start=time.perf_counter()
    print('Verify quality v10',directory,flush=True)
    with (ROOT/'artifacts/quality-v10/offline_verification.log').open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',str(directory/'.development/run_validation_notebook.py'),'Avito.ipynb'],
            cwd=directory,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    execution=json.loads((directory/'artifacts/notebook_execution.json').read_text())
    assert execution['device']==('cuda' if use_gpu else 'cpu') and execution['network_connections_disabled']
    assert output.read_bytes()==reference
    report={'fresh_extraction':True,'existing_answer_removed':True,'original_pool_caches_available':False,
        'device':execution['device'],'network_connections_disabled':True,'answer_bytes_equal':True,
        'answer_sha256':hashlib.sha256(reference).hexdigest(),
        'notebook_code_sha256':execution['code_sha256'],'code_cells_executed':execution['code_cells_executed'],
        'elapsed_seconds':round(time.perf_counter()-start,2),'cross_encoder_device_if_selected':execution['device'],
        'verification_directory':directory.relative_to(ROOT).as_posix()}
    assert report['answer_sha256']==manifest['answer_sha256']
    (ROOT/'artifacts/quality-v10/reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'deliverables/v10/Avito.ipynb').write_bytes((directory/'Avito.ipynb').read_bytes())
    print('Quality v10 reproduced',json.dumps(report),flush=True)


if __name__=='__main__':verify()
