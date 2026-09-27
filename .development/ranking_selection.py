"""Select on known/unknown query regimes, then evaluate reserved audit once."""
from pathlib import Path
source = Path('.development/ranking_experiment.py').read_text(encoding='utf-8')
selection_driver = __file__
__file__ = str(Path('.development/ranking_experiment.py').resolve())
exec(compile(source.split("dataset=grouped_training")[0], str(Path('.development/ranking_experiment.py')), 'exec'), globals())
__file__ = selection_driver
dataset = joblib.load(CACHE/f'expanded_training_{fingerprint}.joblib')

def batched_new_scores(model, features, family, trees=None):
    output=[]
    for start in range(0,len(features),64):
        block=features[start:start+64];x=np.concatenate(block)
        params=({'num_iteration':trees} if family.startswith('lgb') else {'ntree_end':trees}) if trees else {}
        scores=model.predict(x,**params) if family.endswith('rank') else model.predict_proba(x,**params)[:,1]
        output.extend(np.split(scores,np.cumsum([len(a) for a in block])[:-1]))
    return output

def seen_query_history(qframe, truths):
    # Simulate a known text with a held-out context. Own context labels are removed
    # and 90% target IDs are globally purged, but other contexts of the text remain.
    positive_ids=np.array(sorted({ITEM_IDS[i] for truth in truths for i in truth}))
    local=np.random.default_rng(SEED+711)
    cold=set(local.choice(positive_ids,int(.9*len(positive_ids)),replace=False))
    fit=history_all[~history_all.context_key.isin(qframe.context_key)&~history_all.item_id.isin(cold)
        &~history_all.query_norm.isin(new_audit.query_norm)&~history_all.item_id.isin(new_cold_ids)].copy()
    assert not set(qframe.context_key)&set(fit.context_key)
    assert not cold&set(fit.item_id)
    return fit

def strata(frame):
    length=frame.query_norm.str.len().to_numpy()
    return list(zip(np.digitize(length,[15,25]).tolist(),
        (frame.search_infm_params_text.str.len()>0).tolist(),frame.search_location_id.isin(centers.index).tolist()))

def matched_weights(frame, target):
    actual=strata(frame);observed=Counter(actual);desired=Counter(strata(target))
    missing=set(desired)-set(observed)
    assert not missing, ('Missing validation strata',missing)
    weights=np.array([desired.get(k,0)/observed[k]/len(target) for k in actual],dtype=float)
    return weights/weights.sum()

history=HistorySignals(clean_history)
dev_records=cached_hybrid_records(development,history,'new_development')
dev_features=build_rank_feature_records(development,dev_records,'new_development')
reference=prior_scores(dev_records,dev_features)
seen_history=HistorySignals(seen_query_history(development,development_labels))
seen_records=cached_hybrid_records(development,seen_history,'seen_development')
seen_features=build_rank_feature_records(development,seen_records,'seen_development')
seen_reference=prior_scores(seen_records,seen_features)
known=queries.query_norm.isin(history_all.query_norm)
known_fraction=float(known.mean())
unknown_weights=matched_weights(development,queries[~known])
known_weights=matched_weights(development,queries[known])
def selection_metrics(predictions,seen_predictions):
    a=per_query_recall(predictions,development_labels)
    b=per_query_recall(seen_predictions,development_labels)
    return {'development_recall50':float((1-known_fraction)*(a@unknown_weights)+known_fraction*(b@known_weights)),
        'unknown_macro_recall50':float(a.mean()),'known_macro_recall50':float(b.mean())}

ref_metrics=selection_metrics(direct_predict(dev_records,reference),direct_predict(seen_records,seen_reference))
print('V0.3 reference by regime:',ref_metrics,flush=True)
(CACHE/'development_weighting.json').write_text(json.dumps({'known_query_fraction':known_fraction,
    'validation_contexts_per_regime':len(development),'strata':['query length','filters present','location in corpus'],
    'unknown_effective_sample_size':float(1/(unknown_weights**2).sum()),
    'known_effective_sample_size':float(1/(known_weights**2).sum()),'reference':ref_metrics},indent=2),encoding='utf-8')
base_groups=set(expanded_training.index[expanded_training.context_key.isin(training_queries.context_key)])
subset=np.isin(dataset['group'],list(base_groups))
experiments=[{'model':'prior_v3','family':'prior','scale':'4k','trees':120,'weight':0.,**ref_metrics}]
model_paths={}
for scale,mask in [('4k',subset),('expanded',np.ones(len(dataset['y']),dtype=bool))]:
    x,y,w,g=dataset['X'][mask],dataset['y'][mask],dataset['weight'][mask],dataset['group'][mask]
    order=np.argsort(g,kind='stable');x,y,w,g=x[order],y[order],w[order],g[order]
    w /= w.mean()
    _,sizes=np.unique(g,return_counts=True)
    for family in ['lgb_rank','lgb_class','cat_rank','cat_class']:
        name=f'{family}_{scale}';path=CACHE/f'{name}_800_{fingerprint}.joblib';started=time.perf_counter()
        if path.exists():model=joblib.load(path)
        else:
            if family.startswith('lgb'):
                cls=lgb.LGBMRanker if family=='lgb_rank' else lgb.LGBMClassifier
                model=cls(n_estimators=800,num_leaves=31,learning_rate=.05,max_bin=127,min_child_samples=50,
                    reg_lambda=10,random_state=SEED,n_jobs=8,verbosity=-1,deterministic=True,force_col_wise=True,
                    **({'lambdarank_truncation_level':55,'label_gain':[0,1]} if family=='lgb_rank' else {}))
                model.fit(x,y,**({'group':sizes} if family=='lgb_rank' else {'sample_weight':w}))
            else:
                cls=CatBoostRanker if family=='cat_rank' else CatBoostClassifier
                model=cls(iterations=800,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=SEED,
                    thread_count=8,allow_writing_files=False,verbose=100,
                    loss_function='YetiRank:mode=NDCG;top=50' if family=='cat_rank' else 'Logloss')
                model.fit(x,y,**({'group_id':g} if family=='cat_rank' else {'sample_weight':w}))
            save_cache(model,path)
        model_paths[name]=str(path)
        for trees in [200,400,800]:
            scores=batched_new_scores(model,dev_features,family,trees)
            seen_scores=batched_new_scores(model,seen_features,family,trees)
            for weight in [.25,.5,.75,1.]:
                metrics=selection_metrics(combined_predict(dev_records,scores,reference,weight),
                    combined_predict(seen_records,seen_scores,seen_reference,weight))
                experiments.append({'model':name,'family':family,'scale':scale,'trees':trees,'weight':weight,**metrics})
        pd.DataFrame(experiments).sort_values('development_recall50',ascending=False).to_csv(CACHE/'development_experiments_v2.csv',index=False)
        print(name,'best matched dev',max(r['development_recall50'] for r in experiments if r['model']==name),
            'seconds',round(time.perf_counter()-started,1),flush=True)
    del x,y,w,g;gc.collect()
# Require the unknown-query stress check to remain within 0.5pp of the reference.
eligible=[r for r in experiments if r['unknown_macro_recall50']>=ref_metrics['unknown_macro_recall50']-.005]
best=sorted(eligible,key=lambda r:(-r['development_recall50'],r['model'],r['trees'],r['weight']))[0]
(CACHE/'choice_v2.json').write_text(json.dumps({'selected':best,'model_paths':model_paths,'protocol':protocol,
    'config':EXPERIMENT_CONFIG,'known_query_fraction':known_fraction},indent=2),encoding='utf-8')
print('Frozen before reserved audit:',best,flush=True)
test_records=cached_hybrid_records(new_audit,history,'new_audit')
test_features=build_rank_feature_records(new_audit,test_records,'new_audit')
test_reference=prior_scores(test_records,test_features)
if best['family']=='prior':predictions=direct_predict(test_records,test_reference)
else:
    model=joblib.load(model_paths[best['model']])
    scores=batched_new_scores(model,test_features,best['family'],best['trees'])
    predictions=combined_predict(test_records,scores,test_reference,best['weight'])
a=per_query_recall(direct_predict(test_records,test_reference),new_audit_labels)
b=per_query_recall(predictions,new_audit_labels)
local=np.random.default_rng(SEED+707);delta=b-a
boot=[float(local.choice(delta,len(delta),replace=True).mean()) for _ in range(4000)]
result={'queries':len(new_audit),'reference_recall50':float(a.mean()),'selected_recall50':float(b.mean()),
    'delta':float(delta.mean()),'paired_delta_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
    'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),'used_for_selection':False,
    'pool_recall':float(per_query_recall([r[0] for r in test_records],new_audit_labels).mean()),'choice':best}
(CACHE/'audit_metrics_v2.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
pd.DataFrame({'query':new_audit.search_query,'reference':a,'selected':b}).to_csv(CACHE/'audit_comparison_v2.csv',index=False)
print('RESERVED AUDIT',json.dumps(result),flush=True)
