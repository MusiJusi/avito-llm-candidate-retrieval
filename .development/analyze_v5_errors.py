"""Inspect selected predictions and feature gains without fitting or selection."""
import ast
import gc
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.ranking_deps'),str(ROOT/'.inspection_deps')]
sys.stdout.reconfigure(encoding='utf-8')
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)
CACHE=ROOT/'artifacts/ranking-v5'
OLD_CACHE=ROOT/'artifacts/ranking-v1'
selection=json.loads((CACHE/'selection.json').read_text())
winner=selection['winner'];fp=selection['fingerprint']
best_config={'title':.3,'body':.5,'char':.2,'filter':.03,'geo':.5,'micro':.5}
snapshot=ROOT/'Avito_v0.4.ipynb'
nb=json.loads((snapshot if snapshot.exists() else ROOT/'Avito.ipynb').read_text(encoding='utf-8'))
definitions={}
for cell in nb['cells']:
    if cell['cell_type']=='code':
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef):
                definitions[node.name]=node
for name in ['stable_topk','score_candidates','score_lexical_columns','model_scores','per_query_recall']:
    exec(compile(ast.Module(body=[definitions[name]],type_ignores=[]),'<frozen v4 function>','exec'),globals())
legacy_score_candidates=score_candidates
score_candidates=score_lexical_columns
source=(ROOT/'.development/ranking_v5.py').read_text(encoding='utf-8')
for node in ast.parse(source).body:
    if isinstance(node,ast.FunctionDef) and node.name in {'predict_scores','top50','old_v4_scores','blend_scores'}:
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<trained v5 function>','exec'),globals())
frame=pd.read_parquet(ROOT/'artifacts/validation-v5/development_contexts.parquet')
items=pd.read_parquet(ROOT/'benchmark_items.parquet',columns=['item_id','item_location_id']).sort_values('item_id').reset_index(drop=True)
ITEM_IDS=items.item_id.to_numpy(dtype=str)
mapping={value:row for row,value in enumerate(ITEM_IDS)}
truths=[set(mapping[value] for value in text.split()) for text in frame.relevant_item_ids]
model=None if winner['model']=='v4' else joblib.load(CACHE/f"{winner['model']}_{fp}.joblib")
query_rows,pair_rows=[],[]
for mode in ['unseen_text','held_context']:
    records,matrices,known,audit=joblib.load(CACHE/f'development_{mode}_{fp}.joblib')
    baseline=old_v4_scores(records,matrices)
    learned=baseline if model is None else blend_scores(predict_scores(model,matrices,int(winner['trees'])),baseline,float(winner['weight']))
    old_recall=per_query_recall(top50(records,baseline),truths)
    new_recall=per_query_recall(top50(records,learned),truths)
    for position,(q,truth,record,matrix,a,b) in enumerate(zip(frame.itertuples(index=False),truths,records,matrices,baseline,learned)):
        ids=record[0]
        old_order=ids[np.lexsort((ids,-a))];new_order=ids[np.lexsort((ids,-b))]
        old_rank={int(item):rank+1 for rank,item in enumerate(old_order)}
        new_rank={int(item):rank+1 for rank,item in enumerate(new_order)}
        lookup={int(item):row for row,item in enumerate(ids)}
        query_rows.append({'mode':mode,'context_key':q.context_key,'query':q.search_query,
            'actual_known_text':bool(known[position]),'positive_items':len(truth),
            'old_recall50':float(old_recall[position]),'new_recall50':float(new_recall[position]),
            'delta':float(new_recall[position]-old_recall[position]),
            'filters_present':bool(q.search_infm_params_text)})
        for item in sorted(truth):
            row=lookup.get(item)
            pair_rows.append({'mode':mode,'context_key':q.context_key,'query':q.search_query,
                'item_id':ITEM_IDS[item],'old_rank':old_rank.get(item,0),'new_rank':new_rank.get(item,0),
                'same_location':bool(items.item_location_id.iloc[item]==q.search_location_id),
                'distance_km':float(np.expm1(matrix[row,12])) if row is not None and np.isfinite(matrix[row,12]) else None})
    del records,matrices,baseline,learned
    gc.collect()
queries=pd.DataFrame(query_rows)
pairs=pd.DataFrame(pair_rows)
queries.to_csv(CACHE/'development_query_deltas.csv',index=False)
pairs.to_csv(CACHE/'development_positive_ranks.csv',index=False)
summary={}
for mode,group in queries.groupby('mode'):
    summary[mode]={'contexts':len(group),'improved':int((group.delta>0).sum()),
        'worsened':int((group.delta<0).sum()),'macro_delta':float(group.delta.mean())}
(CACHE/'error_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
importance=[]
for name in ['cold_wide','mixed_wide','mixed_mined','final_ranker']:
    path=CACHE/f'{name}_{fp}.joblib'
    if not path.exists():
        continue
    fitted=joblib.load(path)
    gains=fitted.booster_.feature_importance(importance_type='gain')
    for feature,gain in zip(selection['feature_names'],gains):
        importance.append({'model':name,'feature':feature,'gain':float(gain),
                           'gain_fraction':float(gain/max(gains.sum(),1))})
pd.DataFrame(importance).to_csv(CACHE/'feature_importance.csv',index=False)
print(json.dumps(summary,indent=2),flush=True)
