"""Read-only model review: recover existing splits and inspect cached predictions.

No fitting, hyperparameter selection or answer.csv writes. Derived review tables
are kept separately from the submitted ranking-v1 artifacts.
"""
import ast
import hashlib
import html
import json
import re
import sys
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ['.inspection_deps', '.ranking_deps']:
    sys.path.insert(0, str(ROOT / folder))
sys.stdout.reconfigure(encoding='utf-8')
import gc
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)

ORIGINAL = ROOT / 'artifacts/ranking-v1'
CACHE = ROOT / 'artifacts/review-v04'
CACHE.mkdir(exist_ok=True)
nb = json.loads(subprocess.check_output(['git','show','v0.4.0:Avito.ipynb'],cwd=ROOT).decode('utf-8'))
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
definitions = {}
for source in codes:
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
SEED = 260926
CONFIG = {'validation_texts':1400, 'cold_item_fraction':.9, 'development_fraction':.5}
RANKER_CONFIG = {'audit_queries':600, 'fresh_audit_queries':600, 'training_queries':4000}
QUERY_COLS = ['search_query','search_location_id','search_is_delivery_search',
              'search_infm_params_text','search_category']
for name in ['normalize_text','stable_topk','select_query_contexts','query_labels','purge_history',
             'seen_query_history','model_scores','prior_scores','batched_new_scores','score_candidates',
             'score_lexical_columns','per_query_recall']:
    exec(compile(ast.Module(body=[definitions[name]],type_ignores=[]),'<submitted function>','exec'),globals())
legacy_score_candidates = score_candidates
score_candidates = score_lexical_columns
best_config = {'title':.3,'body':.5,'char':.2,'filter':.03,'geo':.5,'micro':.5}
train = pd.read_parquet(ROOT/'train.parquet',columns=QUERY_COLS+[
    'item_id','item_location_id','item_microcat_id','item_latitude','item_longitude'])
queries = pd.read_parquet(ROOT/'benchmark_queries.parquet')
items = pd.read_parquet(ROOT/'benchmark_items.parquet',columns=[
    'item_id','item_location_id','item_microcat_id','item_latitude','item_longitude'])
items = items.sort_values('item_id').reset_index(drop=True)
ITEM_IDS = items.item_id.to_numpy(dtype=str)
ITEM_TO_ROW = {v:i for i,v in enumerate(ITEM_IDS)}
for frame in [train,queries]:
    for column in ['search_query','search_infm_params_text']:
        frame[column] = frame[column].fillna('').astype('string[pyarrow]')
    frame['query_norm'] = frame.search_query.map(normalize_text).astype('string[pyarrow]')
    frame['context_key'] = [hashlib.sha256(json.dumps([str(v) for v in row],ensure_ascii=False).encode()).hexdigest()[:24]
                            for row in frame[QUERY_COLS].itertuples(index=False,name=None)]
for frame in [train,items]:
    for col in ['item_latitude','item_longitude']:
        frame[col]=pd.to_numeric(frame[col],errors='coerce').astype(float)
dedup = train.drop_duplicates(['context_key','item_id'])
eligible = dedup[dedup.item_id.isin(ITEM_TO_ROW)]
item_centers = items.groupby('item_location_id')[['item_latitude','item_longitude']].median().dropna()
train_centers = train.drop_duplicates('item_id').groupby('item_location_id')[['item_latitude','item_longitude']].median().dropna()
missing_center = ~queries.search_location_id.isin(item_centers.index)
recovered_center = missing_center & queries.search_location_id.isin(train_centers.index)
summary = {'data':{
    'raw_train_pairs':len(train),'deduplicated_train_pairs':len(dedup),
    'train_unique_items':int(train.item_id.nunique()),
    'raw_pairs_with_item_in_corpus':int(train.item_id.isin(ITEM_TO_ROW).sum()),
    'deduplicated_pairs_with_item_in_corpus':len(eligible),
    'eligible_normalized_texts':int(eligible.query_norm.nunique()),
    'eligible_contexts':int(eligible.context_key.nunique()),
    'query_contexts_total':int(dedup.context_key.nunique()),
    'benchmark_known_text_fraction':float(queries.query_norm.isin(train.query_norm).mean()),
    'benchmark_known_context_fraction':float(queries.context_key.isin(train.context_key).mean()),
    'benchmark_text_with_corpus_positive_fraction':float(queries.query_norm.isin(eligible.query_norm).mean()),
    'corpus_microcategories':int(items.item_microcat_id.nunique())},
    'geography':{'benchmark_queries_without_corpus_center':int(missing_center.sum()),
                 'recoverable_with_train_item_centers':int(recovered_center.sum())}}
del dedup,eligible
# Reuse the exact historical split recipe. Its small split file is written only
# to review-v04; no production artifact is overwritten.
exec(compile(codes[3],'<historical split>','exec'),globals())
history_source=codes[5]
exec(compile(history_source[:history_source.index('started = time.perf_counter()')],'<history split>','exec'),globals())
new_audit = pd.read_parquet(ORIGINAL/'new_audit_queries.parquet')
new_audit_labels = query_labels(ranker_history,new_audit)
clean_history,new_cold_ids = purge_history(ranker_history,new_audit,new_audit_labels,SEED+703)
development=pd.concat([validation[[*QUERY_COLS,'query_norm','context_key']],audit_queries,fresh_queries],ignore_index=True)
development_labels=labels+audit_labels+fresh_labels
seen_history=seen_query_history(development,development_labels)
full_labels=query_labels(history_all,new_audit)
clean_item_ids=set(clean_history.item_id)
summary['validation']={
    'nominal_seen_regime_actual_known_text_fraction':float(development.query_norm.isin(seen_history.query_norm).mean()),
    'audit_positive_ids':sum(map(len,new_audit_labels)),
    'audit_positive_ids_from_full_labels':sum(map(len,full_labels)),
    'audit_contexts_with_truncated_labels':sum(a!=b for a,b in zip(new_audit_labels,full_labels)),
    'audit_macro_seen_positive_fraction':float(np.mean([sum(ITEM_IDS[i] in clean_item_ids for i in truth)/len(truth) for truth in new_audit_labels]))}
fp=json.loads((ORIGINAL/'manifest.json').read_text())['fingerprint']
records=joblib.load(ORIGINAL/f'new_audit_hybrid_{fp}.joblib')
features=joblib.load(ORIGINAL/f'new_audit_rank_features_{fp}.joblib')
prior_legacy=joblib.load(ROOT/'models/retrieval-priors/evaluation_legacy.joblib')
prior_semantic=joblib.load(ROOT/'models/retrieval-priors/evaluation_semantic.joblib')
reference=prior_scores(records,features)
model=joblib.load(ORIGINAL/f'lgb_rank_expanded_200_{fp}.joblib')
learned=batched_new_scores(model,features,'lgb_rank',200)
combined=[.75/(60+rankdata(-s,method='min'))+.25/(60+rankdata(-p,method='min')) for s,p in zip(learned,reference)]
ordered=[ids[np.lexsort((ids,-score))] for (ids,_),score in zip(records,combined)]
curve={str(k):float(per_query_recall([o[:k] for o in ordered],new_audit_labels).mean())
       for k in [10,20,50,75,100,150,200,300,500,1000]}
assert curve['50']==.9325
summary['audit_recall_curve']=curve
summary['audit_full_label_recall50']=float(per_query_recall([o[:50] for o in ordered],full_labels).mean())
summary['audit_full_label_recall_curve']={str(k):float(per_query_recall([o[:k] for o in ordered],full_labels).mean())
       for k in [10,20,50,75,100,150,200,300,500,1000]}
rows=[]
for q,truth,order,feature,record in zip(new_audit.itertuples(index=False),full_labels,ordered,features,records):
    ranks={int(i):j+1 for j,i in enumerate(order)}
    rowmap={int(i):j for j,i in enumerate(record[0])}
    for item in sorted(truth):
        ix=rowmap.get(item)
        rows.append({'query':q.search_query,'item_id':ITEM_IDS[item],
                     'rank':ranks.get(item,0),'same_location':bool(items.item_location_id.iloc[item]==q.search_location_id),
                     'query_location':q.search_location_id,'item_location':int(items.item_location_id.iloc[item]),
                     'location_center_missing':q.search_location_id not in centers.index,
                     'filter_present':bool(q.search_infm_params_text),
                     'geo_compatibility':float(feature[ix,4]) if ix is not None else None,
                     'micro_compatibility':float(feature[ix,5]) if ix is not None else None})
analysis=pd.DataFrame(rows)
analysis.to_csv(CACHE/'positive_ranks.csv',index=False)
summary['audit_losses']={
    'outside_pool':int(analysis['rank'].eq(0).sum()),
    'rank51_100':int(analysis['rank'].between(51,100).sum()),
    'rank101_300':int(analysis['rank'].between(101,300).sum()),
    'rank301_plus':int(analysis['rank'].gt(300).sum()),
    'same_location_positive_pairs':int(analysis.same_location.sum()),
    'different_location_positive_pairs':int((~analysis.same_location).sum()),
    'same_location_pair_hit50':float(analysis.loc[analysis.same_location,'rank'].between(1,50).mean()),
    'different_location_pair_hit50':float(analysis.loc[~analysis.same_location,'rank'].between(1,50).mean())}
dataset=joblib.load(ORIGINAL/f'expanded_training_{fp}.joblib')
_,sizes=np.unique(dataset['group'],return_counts=True)
summary['training']={'groups':len(sizes),'mean_group_size':float(sizes.mean()),
                     'median_group_size':float(np.median(sizes)),
                     'mean_audit_candidate_pool':float(np.mean([len(r[0]) for r in records]))}
(CACHE/'diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
