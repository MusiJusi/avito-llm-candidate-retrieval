"""Ablate service-specific geography alone and alongside microcat classifiers.

The v5 mined groups and OOF histories are unchanged. The held control is reported
after selection, with its previous-use limitation. No submission is overwritten.
"""
from pathlib import Path
import ast
import hashlib
import json
import time
import gc

GEO_DRIVER=Path(__file__).resolve()
rank_source=GEO_DRIVER.with_name('microcat_rank_v6.py')
tree=ast.parse(rank_source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(rank_source)
exec(compile(tree,str(rank_source),'exec'),globals())
__file__=str(GEO_DRIVER)
from service_geography import ServiceGeography,SERVICE_GEO_FEATURE_NAMES
GEO_CACHE=ROOT/'artifacts/service-geography-v6'
GEO_CACHE.mkdir(exist_ok=True)
GEO_FP=hashlib.sha256((V5_FP+RANK_FP+hashlib.sha256(GEO_DRIVER.read_bytes()).hexdigest()+
    hashlib.sha256((ROOT/'.development/service_geography.py').read_bytes()).hexdigest()).encode()).hexdigest()[:16]


def prepare_service_geo():
    data=joblib.load(V5_CACHE/f'evaluation_mined_{V5_FP}.joblib')
    path=GEO_CACHE/f'oof_features_{GEO_FP}.joblib'
    if USE_CACHE and path.exists():return data,joblib.load(path)
    extra=np.empty((len(data['y']),5),np.float32)
    miner=joblib.load(V5_CACHE/f'mixed_wide_{V5_FP}.joblib')
    random=np.random.default_rng(SEED+811)
    truths=labels_from_gold(gold,training)
    cursor=group=0
    audits=[]
    for mode in ['unseen_text','held_context']:
        assignments=oof_assignments(training,mode,3,SEED+803)
        for fold in range(3):
            positions=np.flatnonzero(assignments==fold)
            frame=training.iloc[positions]
            truth=[truths[i] for i in positions]
            fit,_,_=history_for_queries(training_history,frame,truth,ITEM_IDS,mode,.9,SEED+804+fold)
            geography=ServiceGeography(fit)
            audits.append({'mode':mode,'fold':fold,'history_sha256':history_digest(fit),'own_context_overlap':0})
            assert not set(frame.context_key)&set(fit.context_key)
            del fit
            for start in range(0,len(positions),128):
                records,matrices=joblib.load(V5_CACHE/f'evaluation_pool_{mode}_{fold}_{start}_{V5_FP}.joblib')
                for offset,((ids,base),features) in enumerate(zip(records,matrices)):
                    selected,target=sample_v5_group(ids,base,features,truth[start+offset],random,miner)
                    if not len(selected):continue
                    count=len(selected)
                    assert data['query_position'][group]==positions[start+offset] and data['mode'][group]==mode
                    assert np.array_equal(data['X'][cursor:cursor+count],features[selected],equal_nan=True)
                    assert np.array_equal(data['y'][cursor:cursor+count],target[selected])
                    q=frame.iloc[start+offset]
                    extra[cursor:cursor+count]=geography.features(int(q.search_location_id),ITEM_MICROCATS[ids[selected]],ITEM_LOCS[ids[selected]])
                    cursor+=count;group+=1
                if start%2048==0:print('Service geo OOF',mode,fold,'processed',min(start+128,len(frame)),flush=True)
            del geography,records,matrices
            gc.collect()
    assert cursor==len(data['y']) and group==len(data['sizes'])
    save_cache(extra,path)
    (GEO_CACHE/'oof_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    return data,extra


def fit_service_rank(data,geo,micro,variant):
    path=GEO_CACHE/f'evaluation_{variant}_{GEO_FP}.joblib'
    if USE_CACHE and path.exists():return joblib.load(path)
    x=np.column_stack([data['X'],geo] if variant=='geo' else [data['X'],micro,geo])
    model=lgb.LGBMRanker(n_estimators=400,num_leaves=31,learning_rate=.05,max_bin=127,
        min_child_samples=50,reg_lambda=10,random_state=SEED,n_jobs=8,verbosity=-1,
        deterministic=True,force_col_wise=True,lambdarank_truncation_level=55,label_gain=[0,1])
    started=time.perf_counter()
    print('Fit service geography',variant,x.shape,flush=True)
    model.fit(x,data['y'],group=data['sizes'],sample_weight=np.repeat(data['group_weight'],data['sizes']))
    save_cache(model,path)
    print('Fitted service geography',variant,'seconds',round(time.perf_counter()-started,1),flush=True)
    del x
    gc.collect()
    return model


def service_evaluation(frame,mode,stage):
    other=control if stage=='development' else development
    records,base,micro,known,audit=augmented_evaluation(frame,mode,stage)
    geography=ServiceGeography(evaluation_history(frame,mode,other))
    geo=[geography.features(int(query.search_location_id),ITEM_MICROCATS[ids],ITEM_LOCS[ids])
         for query,(ids,_) in zip(frame.itertuples(index=False),records)]
    return records,base,micro,geo,known,audit


def run_service_experiment():
    data,geo=prepare_service_geo()
    micro=joblib.load(MICRO_CACHE/f'evaluation_oof_features_{RANK_FP}.joblib')
    models={name:fit_service_rank(data,geo,micro,name) for name in ['geo','micro_geo']}
    del data,geo,micro
    gc.collect()
    bundles={mode:service_evaluation(development,mode,'development') for mode in ['unseen_text','held_context']}
    baseline_scores={mode:joblib.load(MICRO_CACHE/f'development_{mode}_v5scores_{MICRO_FP}.joblib') for mode in bundles}
    a=bundles['unseen_text'];b=bundles['held_context']
    baseline=development_metrics(top50(a[0],baseline_scores['unseen_text']),top50(b[0],baseline_scores['held_context']),b[4])
    rows=[{'variant':'v5','trees':400,'weight':0.,**baseline}]
    micro_selection=json.loads((MICRO_CACHE/'ranker_selection.json').read_text())
    rows.append({'variant':'micro','trees':micro_selection['winner']['trees'],'weight':micro_selection['winner']['weight'],
                 **{key:micro_selection['winner'][key] for key in baseline}})
    for variant,model in models.items():
        for trees in [200,400]:
            scores={}
            for mode,bundle in bundles.items():
                matrix=[np.column_stack([base,geo] if variant=='geo' else [micro,geo])
                        for base,micro,geo in zip(bundle[1],bundle[2],bundle[3])]
                scores[mode]=predict_scores(model,matrix,trees)
                del matrix
            for weight in [.5,.75,1.]:
                metrics=development_metrics(top50(a[0],blend_scores(scores['unseen_text'],baseline_scores['unseen_text'],weight)),
                    top50(b[0],blend_scores(scores['held_context'],baseline_scores['held_context'],weight)),b[4])
                rows.append({'variant':variant,'trees':trees,'weight':weight,**metrics})
            print('Service development',variant,trees,'best',max(r['matched_recall50'] for r in rows if r['variant']==variant),flush=True)
    table=pd.DataFrame(rows).sort_values(['matched_recall50','variant','trees','weight'],ascending=[False,True,True,True])
    table.to_csv(GEO_CACHE/'ablation.csv',index=False,lineterminator='\n')
    winner=table[table.unseen_macro_recall50>=baseline['unseen_macro_recall50']-.002].iloc[0].to_dict()
    selection={'winner':winner,'baseline':baseline,'fingerprint':GEO_FP,'control_used_for_selection':False,
        'feature_names':SERVICE_GEO_FEATURE_NAMES,'base_micro_selection':micro_selection['winner']}
    (GEO_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('SERVICE SELECTED',json.dumps(selection),flush=True)
    del bundles,a,b,baseline_scores
    gc.collect()
    records,base,micro,geo,_,audit=service_evaluation(control,'unseen_text','control')
    original=baseline_v5(records,base)
    if winner['variant']=='v5':
        chosen=original
    elif winner['variant']=='micro':
        model=joblib.load(MICRO_CACHE/f'evaluation_rank_both_{RANK_FP}.joblib')
        chosen=blend_scores(predict_scores(model,micro,int(winner['trees'])),original,float(winner['weight']))
    else:
        matrix=[np.column_stack([x,z] if winner['variant']=='geo' else [y,z]) for x,y,z in zip(base,micro,geo)]
        chosen=blend_scores(predict_scores(models[winner['variant']],matrix,int(winner['trees'])),original,float(winner['weight']))
    truth=labels_from_gold(gold,control)
    before=per_query_recall(top50(records,original),truth);after=per_query_recall(top50(records,chosen),truth)
    delta=after-before
    random=np.random.default_rng(SEED+906)
    boot=[float(random.choice(delta,len(delta),replace=True).mean()) for _ in range(4000)]
    report={'v5_recall50':float(before.mean()),'selected_recall50':float(after.mean()),'delta':float(delta.mean()),
        'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),
        'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'used_for_selection':False,
        'limitation':'Control previously viewed in v5 and microcat experiments; frozen v4 auxiliary-history limitation.',
        'history_audit':audit,'answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH}
    assert report['answer_unchanged']
    (GEO_CACHE/'control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SERVICE CONTROL',json.dumps(report),flush=True)


if __name__=='__main__':
    run_service_experiment()
