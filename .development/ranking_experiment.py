"""Controlled ranking experiment; submission export is a separate final step."""
from pathlib import Path
import sys, json, hashlib, os, time, gc
sys.path[:0] = [str(Path('.ranking_deps').resolve()), str(Path('.inspection_deps').resolve()), str(Path('.semantic_deps').resolve())]
from bootstrap_ranking import bootstrap
bootstrap(globals())
from catboost import CatBoostRanker, CatBoostClassifier
import lightgbm as lgb

PRIOR_CACHE = CACHE
PRIOR_FINGERPRINT = fingerprint
prior_legacy = joblib.load(PRIOR_CACHE / f'hgb_leaves15_trees120_{fingerprint}.joblib')
prior_semantic = joblib.load(PRIOR_CACHE / f'semantic_hgb_leaves15_trees120_{fingerprint}.joblib')
CACHE = ROOT / 'artifacts' / 'ranking-v1'
CACHE.mkdir(exist_ok=True)
EXPERIMENT_CONFIG = {'contexts': 20000, 'contexts_per_text': 4, 'audit_texts': 800,
    'hard_source': 40, 'hard_model': 80, 'random_tail': 64, 'folds': 3, 'seed': SEED + 701}
fingerprint = hashlib.sha256(json.dumps({'inputs':input_hashes,'config':EXPERIMENT_CONFIG,
    'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()[:16]

def context_sample(history, count, seed, include=None):
    eligible = history[history.item_id.isin(ITEM_TO_ROW)].drop_duplicates('context_key')
    local = np.random.default_rng(seed)
    eligible = eligible.sort_values('context_key').iloc[local.permutation(len(eligible))]
    bounded = eligible.groupby('query_norm', sort=False).head(EXPERIMENT_CONFIG['contexts_per_text'])
    if include is not None:
        bounded = pd.concat([include, bounded], ignore_index=True).drop_duplicates('context_key')
    return bounded.head(count)[[*QUERY_COLS,'query_norm','context_key']].sort_values('context_key').reset_index(drop=True)

# Reserve genuinely new texts before any new fit/selection. Old supervised
# positive IDs are excluded from this audit so their labels cannot be reused.
old_direct_texts = set(training_queries.query_norm) | set(original_training_queries.query_norm)
old_direct_ids = {ITEM_IDS[i] for truth in query_labels(ranker_history, training_queries) for i in truth}
overlap_texts = set(ranker_history.loc[ranker_history.item_id.isin(old_direct_ids), 'query_norm'])
audit_source = ranker_history[~ranker_history.query_norm.isin(old_direct_texts | overlap_texts)]
new_audit = select_query_contexts(audit_source, EXPERIMENT_CONFIG['audit_texts'], SEED + 702)
new_audit_labels = query_labels(ranker_history, new_audit)
clean_history, new_cold_ids = purge_history(ranker_history, new_audit, new_audit_labels, SEED + 703)
expanded_training = context_sample(clean_history, EXPERIMENT_CONFIG['contexts'], SEED + 704, training_queries)
assert not set(expanded_training.query_norm) & set(new_audit.query_norm)
assert not set(clean_history.query_norm) & set(new_audit.query_norm)
assert not new_cold_ids & set(clean_history.item_id)
assert not old_direct_ids & {ITEM_IDS[i] for truth in new_audit_labels for i in truth}
development = pd.concat([validation[[*QUERY_COLS,'query_norm','context_key']], audit_queries, fresh_queries],ignore_index=True)
development_labels = labels + audit_labels + fresh_labels
protocol = {'training_contexts':len(expanded_training),'training_texts':expanded_training.query_norm.nunique(),
    'development_contexts':len(development),'new_audit_texts':len(new_audit),
    'direct_training_text_overlap':0,'direct_training_positive_item_overlap':0,'purged_item_overlap':0}
print('New protocol:',json.dumps(protocol),flush=True)
(CACHE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
expanded_training.to_parquet(CACHE/'training_queries.parquet',index=False)
new_audit.to_parquet(CACHE/'new_audit_queries.parquet',index=False)
source_item_path = semantic_index.item_path
destination_item_path = CACHE/source_item_path.name
if not destination_item_path.exists():
    os.link(source_item_path,destination_item_path)
semantic_index.prepare(pd.concat([expanded_training.query_norm,development.query_norm,new_audit.query_norm,queries.query_norm]))

def prior_scores(records, matrices):
    a=model_scores(prior_legacy,[x[:,:32] for x in matrices])
    b=model_scores(prior_semantic,[x[:,:38] for x in matrices])
    output=[]
    for (_,features),legacy,semantic in zip(records,a,b):
        lex=score_candidates(features,best_config)
        prior=.5/(60+rankdata(-legacy,method='min'))+.5/(60+rankdata(-lex,method='min'))
        output.append(.25/(60+rankdata(-semantic,method='min'))+.75/(60+rankdata(-prior,method='min')))
    return output

def direct_predict(records,scores):
    return [ids[stable_topk(score,50)] for (ids,_),score in zip(records,scores)]

def combined_predict(records,scores,reference,weight):
    if weight==0:return direct_predict(records,reference)
    if weight==1:return direct_predict(records,scores)
    return direct_predict(records,[weight/(60+rankdata(-s,method='min'))+(1-weight)/(60+rankdata(-p,method='min')) for s,p in zip(scores,reference)])

def grouped_training(history,qframe,stage):
    path=CACHE/f'{stage}_{fingerprint}.joblib'
    if path.exists():return joblib.load(path)
    truths=query_labels(history,qframe)
    assert all(truths)
    local=np.random.default_rng(SEED+705)
    texts=np.sort(qframe.query_norm.unique())
    assignment={text:int(fold) for text,fold in zip(texts[local.permutation(len(texts))],np.arange(len(texts))%3)}
    xs,ys,weights,groups,audits=[],[],[],[],[]
    for fold in range(3):
        positions=np.flatnonzero(qframe.query_norm.map(assignment).to_numpy()==fold)
        fold_queries=qframe.iloc[positions]
        fold_truths=[truths[i] for i in positions]
        fit,cold=purge_history(history,fold_queries,fold_truths,SEED+706+fold)
        h=HistorySignals(fit)
        print(stage,'fold',fold,'contexts',len(positions),'texts',fold_queries.query_norm.nunique(),flush=True)
        for start in range(0,len(positions),256):
            chunk=fold_queries.iloc[start:start+256]
            records=retrieve_semantic_features(chunk,h,progress_every=1000)
            matrices=[rank_semantic_features(q,r) for q,r in zip(chunk.itertuples(index=False),records)]
            mined=model_scores(prior_legacy,[x[:,:32] for x in matrices])
            for j,((ids,base),features,model_score) in enumerate(zip(records,matrices,mined)):
                truth=fold_truths[start+j]
                target=np.isin(ids,list(truth)); pos=np.flatnonzero(target); neg=np.flatnonzero(~target)
                if not len(pos) or not len(neg):continue
                parts=[neg[stable_topk(score_candidates(base[neg],best_config),EXPERIMENT_CONFIG['hard_source'])],
                    neg[stable_topk(base[neg,7],EXPERIMENT_CONFIG['hard_source'])],
                    neg[stable_topk(model_score[neg],EXPERIMENT_CONFIG['hard_model'])]]
                hard=np.unique(np.concatenate(parts)); tail=np.setdiff1d(neg,hard)
                random=local.choice(tail,min(len(tail),EXPERIMENT_CONFIG['random_tail']),replace=False)
                selected=np.sort(np.concatenate([pos,hard,random])); y=target[selected].astype(np.uint8)
                assert not set(ids[selected[~y.astype(bool)]]) & truth
                xs.append(features[selected]);ys.append(y)
                weights.append(np.where(y,100/max(int(y.sum()),1),100/max(int((1-y).sum()),1)).astype(np.float32))
                groups.append(np.full(len(selected),positions[start+j],dtype=np.int32))
            if start%1024==0:print(stage,'fold',fold,'processed',start+len(chunk),flush=True)
        audits.append({'fold':fold,'contexts':len(positions),'texts':fold_queries.query_norm.nunique(),
            'text_overlap':len(set(fold_queries.query_norm)&set(fit.query_norm)), 'purged_item_overlap':len(cold&set(fit.item_id))})
        del h,fit,records,matrices;gc.collect()
    data={'X':np.concatenate(xs),'y':np.concatenate(ys),'weight':np.concatenate(weights),'group':np.concatenate(groups),'audit':audits}
    data['weight']/=data['weight'].mean()
    save_cache(data,path)
    (CACHE/f'{stage}_leakage_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    print('Training matrix',data['X'].shape,'positives',int(data['y'].sum()),flush=True)
    return data

dataset=grouped_training(clean_history,expanded_training,'expanded_training')
history=HistorySignals(clean_history)
dev_records=cached_hybrid_records(development,history,'new_development')
dev_features=build_rank_feature_records(development,dev_records,'new_development')
reference=prior_scores(dev_records,dev_features)
print('Frozen v0.3 development:',float(per_query_recall(direct_predict(dev_records,reference),development_labels).mean()),flush=True)
base_groups=set(expanded_training.index[expanded_training.context_key.isin(training_queries.context_key)])
subset=np.isin(dataset['group'],list(base_groups))
experiments=[];model_paths={}
for scale,mask in [('4k',subset),('expanded',np.ones(len(dataset['y']),dtype=bool))]:
    x,y,w,g=dataset['X'][mask],dataset['y'][mask],dataset['weight'][mask],dataset['group'][mask]
    _,sizes=np.unique(g,return_counts=True)
    # Sorting keeps query groups contiguous and gives all libraries the same data.
    order=np.argsort(g,kind='stable');x,y,w,g=x[order],y[order],w[order],g[order]
    for family in ['lgb_rank','lgb_class','cat_rank','cat_class']:
        name=f'{family}_{scale}'
        path=CACHE/f'{name}_{fingerprint}.joblib';started=time.perf_counter()
        if path.exists():model=joblib.load(path)
        else:
            if family.startswith('lgb'):
                cls=lgb.LGBMRanker if family=='lgb_rank' else lgb.LGBMClassifier
                model=cls(n_estimators=500,num_leaves=31,learning_rate=.05,max_bin=127,
                    min_child_samples=50,reg_lambda=10,random_state=SEED,n_jobs=8,
                    verbosity=-1,deterministic=True,force_col_wise=True,
                    **({'lambdarank_truncation_level':55,'label_gain':[0,1]} if family=='lgb_rank' else {}))
                model.fit(x,y,**({'group':sizes} if family=='lgb_rank' else {'sample_weight':w}))
            else:
                cls=CatBoostRanker if family=='cat_rank' else CatBoostClassifier
                model=cls(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=10,
                    random_seed=SEED,thread_count=8,allow_writing_files=False,verbose=100,
                    loss_function='YetiRank:mode=NDCG;top=50' if family=='cat_rank' else 'Logloss')
                model.fit(x,y,**({'group_id':g} if family=='cat_rank' else {'sample_weight':w}))
            save_cache(model,path)
        model_paths[name]=str(path)
        scores=[model.predict(a) if family.endswith('rank') else model.predict_proba(a)[:,1] for a in dev_features]
        for weight in [.25,.5,.75,1.]:
            value=float(per_query_recall(combined_predict(dev_records,scores,reference,weight),development_labels).mean())
            experiments.append({'model':name,'weight':weight,'development_recall50':value,'fit_seconds':round(time.perf_counter()-started,2)})
        pd.DataFrame(experiments).sort_values('development_recall50',ascending=False).to_csv(CACHE/'development_experiments.csv',index=False)
        print(name,'best dev',max(r['development_recall50'] for r in experiments if r['model']==name),'seconds',round(time.perf_counter()-started,1),flush=True)
    del x,y,w,g;gc.collect()
best=sorted(experiments,key=lambda r:(-r['development_recall50'],r['model'],r['weight']))[0]
(CACHE/'choice.json').write_text(json.dumps({'selected':best,'model_paths':model_paths,'protocol':protocol,'config':EXPERIMENT_CONFIG},indent=2),encoding='utf-8')
print('Selected before new audit:',best,flush=True)
# This audit is evaluated only after selection; it does not change parameters.
test_records=cached_hybrid_records(new_audit,history,'new_audit')
test_features=build_rank_feature_records(new_audit,test_records,'new_audit')
test_reference=prior_scores(test_records,test_features)
model=joblib.load(model_paths[best['model']]);family=best['model'].split('_')[1]
test_scores=[model.predict(a) if family=='rank' else model.predict_proba(a)[:,1] for a in test_features]
a=per_query_recall(direct_predict(test_records,test_reference),new_audit_labels)
b=per_query_recall(combined_predict(test_records,test_scores,test_reference,best['weight']),new_audit_labels)
local=np.random.default_rng(SEED+707);delta=b-a
boot=[float(local.choice(delta,len(delta),replace=True).mean()) for _ in range(4000)]
result={'queries':len(new_audit),'reference_recall50':float(a.mean()),'selected_recall50':float(b.mean()),
    'delta':float(delta.mean()),'paired_delta_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
    'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),'used_for_selection':False,
    'pool_recall':float(per_query_recall([r[0] for r in test_records],new_audit_labels).mean())}
(CACHE/'audit_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('NEW AUDIT',json.dumps(result),flush=True)
