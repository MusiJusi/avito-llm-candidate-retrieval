"""Package the preserved v5 and separate microcat candidate, omitting large pools."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/microcat-v6'
manifest=json.loads((CACHE/'manifest.json').read_text())
assert hashlib.sha256((ROOT/manifest['answer_file']).read_bytes()).hexdigest()==manifest['answer_sha256']
paths=[ROOT/name for name in ['Avito.ipynb','Avito_microcat_v6.ipynb','Avito_microcat_candidate_v6.ipynb',
    'Avito_v0.3.ipynb','Avito_v0.4.ipynb','Avito_v0.5.ipynb','Avito_validation_v5.ipynb',
    'README.md','README_microcat_v6.md','README_v0.5.md','README_v0.3.md','requirements.txt','CHANGELOG.md',
    'IMPROVEMENT_PLAN.md','IMPROVEMENT_PROGRESS.md','Avito_neural_experiments.ipynb','answer.csv','answer_microcat_v6.csv','answer_v0.4.csv','answer_v0.5.csv']]
paths+=[p for p in (ROOT/'models').rglob('*') if 'mmarco-miniLM-cross-encoder' not in p.parts]
paths+=list((ROOT/'.development').glob('*.py'))
paths+=[p for p in CACHE.iterdir() if p.suffix in {'.json','.csv'} or p.name in
    [manifest['final_ranker'],manifest['input_vectors'],*manifest['final_classifiers'].values()]]
old=ROOT/'artifacts/ranking-v5'
paths+=[p for p in old.iterdir() if p.suffix in {'.json','.csv'} or p.name.startswith(('e5_items_','e5_queries_'))
    or p.name in ['final_ranker_e2854ae5a86a1bbf.joblib','mixed_wide_e2854ae5a86a1bbf.joblib','mixed_mined_e2854ae5a86a1bbf.joblib']]
paths+=[ROOT/'artifacts/ranking-v1'/name for name in ['final_ranker_c0442eeba2d244f2.joblib','lgb_rank_expanded_200_c0442eeba2d244f2.joblib']]
for folder in ['cross-encoder-pilot','query-encoder-pilot','query-encoder-rank','query-micro-blend','service-geography-v6','cold-sensitivity-v5']:
    target=ROOT/'artifacts'/folder
    if target.exists():paths+=[p for p in target.iterdir() if p.suffix in {'.json','.csv'}]
paths=sorted({p for p in paths if p.is_file() and not p.name.endswith('.tmp')})
output=ROOT/'deliverables/avito_v6_experiments.zip'
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
    for path in paths:archive.write(path,path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert archive.read(manifest['answer_file'])==(ROOT/manifest['answer_file']).read_bytes()
    assert archive.read('answer.csv')==(ROOT/'answer.csv').read_bytes()
print('Packaged',output,'MiB',round(output.stat().st_size/2**20,1),flush=True)
