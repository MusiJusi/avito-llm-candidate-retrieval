"""Measure Recall sensitivity to permitted feature-history availability.

Weights/query encoders stay fixed. Histories, first-stage pools and every new
history signal are recomputed for each regime. This tests inference-history
availability, not a retraining experiment or a known law of the hidden test.
The v4 auxiliary-prior limitation remains explicit.
"""
from pathlib import Path
import ast
import gc
import json
import hashlib

SENSITIVITY_DRIVER=Path(__file__).resolve()
source=SENSITIVITY_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(SENSITIVITY_DRIVER)
from activate_warm_history import activate
activate(globals())


def history_at_fraction(frame,mode,fraction):
    # Keep every other exclusion identical to the frozen evaluation recipe.
    original=evaluation_history(frame,mode,control)
    _,cold,_=history_for_queries(history_all,frame,labels_from_gold(gold,frame),ITEM_IDS,mode,.9,SEED+801)
    other_truth=labels_from_gold(gold,control)
    other_ids=np.array(sorted({ITEM_IDS[i] for t in other_truth for i in t}),dtype=str)
    other_cold=set(np.random.default_rng(SEED+812).choice(other_ids,int(.9*len(other_ids)),replace=False))
    all_positive=set(ITEM_IDS[i] for t in labels_from_gold(gold,frame) for i in t)
    if fraction==.9:return original
    if fraction==1.:blocked=all_positive
    elif fraction==0.:blocked=set()
    else:
        # Nested subsets reduce random-membership noise between scenarios.
        blocked=set(np.random.default_rng(SEED+1911).choice(sorted(cold),int(fraction*len(all_positive)),replace=False))
    history,_,_=history_for_queries(history_all,frame,labels_from_gold(gold,frame),ITEM_IDS,mode,0.,SEED+801,
        excluded_texts=control.query_norm,excluded_items=other_cold|blocked)
    assert not set(frame.context_key)&set(history.context_key)
    return history


def sensitivity():
    global RESOURCES,CONTEXT_VECTORS,LEARNED_LOOKUP,QUALITY_BANK
    install_retriever();RESOURCES=prepare_resources()
    CONTEXT_VECTORS,_=contextual_vectors()
    selection=json.loads((V10_CACHE/'selection.json').read_text());winner=selection['winner']
    model=joblib.load(V10_CACHE/f'evaluation_{winner["variant"]}_{V10_FP}.joblib') if winner['variant'] in MODEL_RECIPES else None
    warm_models={};warm_weights={}
    for name,path in [('warm_mixed',ROOT/'artifacts/warm-history-v10/development.csv'),
                      ('warm_deeper',V10_CACHE/'warm_deeper_development.csv')]:
        model_path=V10_CACHE/f'evaluation_{name}_{V10_FP}.joblib'
        if model_path.exists() and path.exists():
            row=pd.read_csv(path).query('variant == @name').sort_values('matched_recall50',ascending=False).iloc[0]
            warm_models[name]=joblib.load(model_path);warm_weights[name]=float(row.weight)
    v7=joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    v9=joblib.load(V9_CACHE/f'evaluation_combined_{V9_FP}.joblib')
    vectors=evaluation_query_features(development,'development')
    LEARNED_LOOKUP=dict(zip(sorted(set(development.query_norm)),vectors))
    output=[];all_audits=[]
    for fraction in [0.,.5,.9,1.]:
        values={};known=None;audits=[];pools=[]
        for mode in ['unseen_text','held_context']:
            recipe_digest=hashlib.sha256((sha256_file(SENSITIVITY_DRIVER)+json.dumps(warm_weights,sort_keys=True)+
                json.dumps(winner,sort_keys=True)).encode()).hexdigest()[:12]
            path=V10_CACHE/f'sensitivity_{mode}_{fraction:g}_{V10_FP}_{recipe_digest}.joblib'
            if path.exists():values[mode],known,audit,pool=joblib.load(path)
            else:
                history=history_at_fraction(development,mode,fraction)
                known=development.query_norm.isin(history.query_norm).to_numpy()
                audit=history_coverage(development,labels_from_gold(gold,development),history,ITEM_IDS)
                QUALITY_BANK=(QualityEvidenceWithTrace if winner['variant'].startswith('warm_') else QualityEvidence)(history)
                trace=None if winner['variant'].startswith('warm_') else QueryItemEvidence(history)
                neighbors=evidence(history,development);retriever_history=V5History(history)
                del history;gc.collect()
                records=retrieve_context_features(development,retriever_history,progress_every=1000)
                features=[v5_features(q,r,retriever_history) for q,r in zip(development.itertuples(index=False),records)]
                base=matrices(development,records,features,neighbors)
                extra=extra_matrices(development,records,base)
                for q,x in zip(development.itertuples(index=False),extra):
                    if float(q.search_category)==0:x[:,70]=1.
                baseline=baseline_v5(records,features)
                reference=blend_scores(predict_scores(v7,[x[:,:51] for x in base],400),baseline,.75)
                reference=blend_scores(predict_scores(v9,base,400),reference,.25)
                chosen=compatibility_boost(development,records,reference,selection['category_boost'])
                if model is not None:
                    recipe=MODEL_RECIPES[winner['variant']]
                    score=predict_scores(model,[x[:,recipe['columns']] for x in extra],recipe['trees'])
                    chosen=blend_scores(score,chosen,winner['weight'])
                    del score
                truth=labels_from_gold(gold,development)
                values[mode]={'v9':per_query_recall(top50(records,reference),truth),
                    'v10':per_query_recall(top50(records,chosen),truth)}
                warm_features=extra if trace is None else [
                    np.column_stack([x,trace.features(q,ITEM_IDS[ids])]).astype(np.float32)
                    for q,(ids,_),x in zip(development.itertuples(index=False),records,extra)]
                for name,warm_model in warm_models.items():
                    recipe=MODEL_RECIPES[name]
                    score=predict_scores(warm_model,warm_features,recipe['trees'])
                    warm_score=blend_scores(score,compatibility_boost(development,records,reference,selection['category_boost']),warm_weights[name])
                    values[mode][name]=per_query_recall(top50(records,warm_score),truth)
                    del score,warm_score
                pool=float(per_query_recall([r[0] for r in records],truth).mean())
                save_cache((values[mode],known,audit,pool),path)
                del records,features,base,extra,warm_features,trace,neighbors,retriever_history,baseline,reference,chosen;gc.collect()
            audits.append(dict(mode=mode,**audit));pools.append(pool)
        for name in ['v9','v10',*warm_models]:
            metrics=matched_metrics(values['unseen_text'][name],values['held_context'][name],known)
            output.append(dict(cold_fraction=fraction,version=name,**metrics,
                pool_unseen=pools[0],pool_held=pools[1]))
        all_audits.extend(dict(cold_fraction=fraction,**audit) for audit in audits)
        report={'rows':output,'audits':all_audits,'weights_fixed':True,'encoders_fixed':True,'warm_weights':warm_weights,
            'source_sha256':sha256_file(SENSITIVITY_DRIVER),
            'limitation':'Inference-history sensitivity, not encoder/ranker retraining or hidden-test history inference.'}
        (V10_CACHE/'history_sensitivity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        pd.DataFrame(output).to_csv(V10_CACHE/'history_sensitivity.csv',index=False,lineterminator='\n')
        print('History sensitivity',pd.DataFrame(output).to_string(index=False),flush=True)


if __name__=='__main__':
    sensitivity()
    # Encoding is independent of history diagnostics, but final selection waits
    # for its bounded pilot. This also resumes an early download-timeout safely.
    from finish_bge_v10 import finish
    finish()
