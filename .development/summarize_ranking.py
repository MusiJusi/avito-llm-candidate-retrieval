"""Describe the final model and training data without changing predictions."""
import ast,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for name in ['.inspection_deps','.ranking_deps']:
    sys.path.insert(0,str(root/name))
import joblib
import numpy as np
import pandas as pd
cache=root/'artifacts/ranking-v1'
manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
nb=json.loads((root/'Avito.ipynb').read_text(encoding='utf-8'))
names={}
for cell in nb['cells']:
    if cell['cell_type']!='code':continue
    for node in ast.parse(''.join(cell['source'])).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.List):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in {'RANK_FEATURE_NAMES','SEMANTIC_FEATURE_NAMES'}:
                    names[target.id]=ast.literal_eval(node.value)
features=names['RANK_FEATURE_NAMES']+names['SEMANTIC_FEATURE_NAMES']
assert len(features)==38
(cache/'feature_names.json').write_text(json.dumps(features,indent=2),encoding='utf-8')
model=joblib.load(cache/manifest['final_model_files'][0])
importance=pd.DataFrame({'feature':features,
    'split_count':model.booster_.feature_importance(importance_type='split'),
    'gain':model.booster_.feature_importance(importance_type='gain')})
importance.sort_values('gain',ascending=False).to_csv(cache/'final_feature_importance.csv',index=False)
stats={}
for stage in ['expanded_training','final_training']:
    data=joblib.load(cache/f"{stage}_{manifest['fingerprint']}.joblib")
    stats[stage]={'rows':len(data['y']),'features':data['X'].shape[1],
        'retrieved_positive_rows':int(data['y'].sum()),'groups':len(np.unique(data['group'])),
        'oof_leakage_audit':data['audit']}
    del data
(cache/'training_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(importance.sort_values('gain',ascending=False).head(8).to_string(index=False))
print(json.dumps(stats,indent=2))
