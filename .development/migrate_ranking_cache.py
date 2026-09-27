"""Reuse controlled-experiment caches after packaging identical numerical code.

Only cache addressing and the USE_CACHE guard changed; a subsequent cache-free
reproduction check verifies the same metrics/submission from the submitted code.
"""
import ast, hashlib, json, os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'.development/notebook_source.py').read_text(encoding='utf-8')
namespace={'SEED':260926}
for node in ast.parse(source).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name in ['CONFIG','SEMANTIC_CONFIG','EXPERIMENT_CONFIG']:
            namespace[name]=eval(compile(ast.Expression(node.value),'<own configuration>','eval'),{},namespace)
prior=json.loads((root/'models/retrieval-priors/manifest.json').read_text())
encoder=json.loads((root/'models/multilingual-e5-small/manifest.json').read_text())
payload={'inputs':prior['input_sha256'],'config':namespace['EXPERIMENT_CONFIG'],
    'baseline_config':namespace['CONFIG'],'semantic':namespace['SEMANTIC_CONFIG'],
    'model':encoder,'priors':prior['sha256'],'data_revision':1}
key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
origin='2ac41b3e7594e9c5'
directory=root/'artifacts/ranking-v1'
for path in sorted(directory.glob(f'*_{origin}.joblib')):
    target=directory/path.name.replace(origin,key)
    assert target.resolve().is_relative_to(directory.resolve())
    if not target.exists():os.link(path,target)
    print(path.name,'->',target.name)
print('Numerical cache key:',key)
