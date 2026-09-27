"""Preserve the exact supervised priors and their complete training recipe."""
from pathlib import Path
import hashlib, json, shutil, subprocess
root=Path(__file__).resolve().parents[1]
snapshot=root/'Avito_v0.3.ipynb'
if not snapshot.exists():
    # The current main notebook may already contain ranking changes.
    snapshot.write_bytes(subprocess.check_output(['git','show','v0.3.0:Avito.ipynb'],cwd=root))
reference=json.loads((root/'artifacts/semantic-v1/manifest.json').read_text(encoding='utf-8'))
key=reference['fingerprint']
mapping={'evaluation_legacy.joblib':f'hgb_leaves15_trees120_{key}.joblib',
    'evaluation_semantic.joblib':f'semantic_hgb_leaves15_trees120_{key}.joblib',
    'final_legacy.joblib':f'final_legacy_ranker_{key}.joblib',
    'final_semantic.joblib':f'final_ranker_{key}.joblib'}
destination=root/'models/retrieval-priors'
destination.mkdir(exist_ok=True)
checksums={}
for name,source in mapping.items():
    origin=root/'artifacts/semantic-v1'/source
    target=destination/name
    shutil.copyfile(origin,target)
    checksums[name]=hashlib.sha256(target.read_bytes()).hexdigest()
manifest={'version':'v0.3.0','source_commit':'ba3fa90cf0878dc440ed745db2af90b060c03464',
    'training_recipe':'Avito_v0.3.ipynb','input_sha256':reference['input_sha256'],
    'evaluation_training_texts':4000,'final_training_texts':6600,'sha256':checksums,
    'purpose':'Frozen negative-mining and ensemble priors; new models are trained in Avito.ipynb.'}
(destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
