"""Package v5 weights and code for local offline reproduction; omit search caches."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/ranking-v5'
manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()==manifest['answer_sha256']
destination=ROOT/'deliverables/avito_v5_solution.zip'
destination.parent.mkdir(exist_ok=True)
paths=[ROOT/name for name in ['Avito.ipynb','Avito_training_v5.ipynb','Avito_validation_v5.ipynb',
    'Avito_v0.3.ipynb','Avito_v0.4.ipynb','README.md','README_v0.3.md','requirements.txt',
    'CHANGELOG.md','IMPROVEMENT_PLAN.md','answer.csv','answer_v0.4.csv']]
paths+=list((ROOT/'models').rglob('*'))
fp=manifest['fingerprint']
paths+=[path for path in CACHE.iterdir() if path.suffix in {'.json','.csv'}
    or path.name in [f'final_ranker_{fp}.joblib',f'mixed_wide_{fp}.joblib']
    or path.name.startswith(('e5_items_','e5_queries_'))]
paths+=[ROOT/'artifacts/ranking-v1'/name for name in [
    'final_ranker_c0442eeba2d244f2.joblib','lgb_rank_expanded_200_c0442eeba2d244f2.joblib']]
paths+=[ROOT/'.development'/name for name in ['bootstrap_ranking.py','ranking_v5.py',
    'validation_protocol.py','run_validation_notebook.py','check_v5_training.py',
    'check_validation_protocol.py','build_validation_notebook.py','build_v5_training_notebook.py',
    'freeze_v5_notebook.py','download_semantic_model.py','validate_submission.py',
    'verify_v5_portable.py','verify_v5_refit.py','validate_v5_submission.py','write_v5_docs.py','analyze_v5_errors.py']]
paths=sorted({path for path in paths if path.is_file() and not path.name.endswith('.tmp')})
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
    for path in paths:
        print('Packaging',path.relative_to(ROOT),flush=True)
        archive.write(path,path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert archive.read('answer.csv')==(ROOT/'answer.csv').read_bytes()
    assert 'models/multilingual-e5-small/model.safetensors' in archive.namelist()
    assert f'artifacts/ranking-v5/final_ranker_{fp}.joblib' in archive.namelist()
print('Ready',destination,'MiB',round(destination.stat().st_size/2**20,1),flush=True)
