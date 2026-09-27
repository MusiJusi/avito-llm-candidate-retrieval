"""Actual Recall sensitivity, separate from mere history-coverage diagnostics.

Same 400 development contexts and frozen v5 model across 0/50/90/100% removal.
The full 3400-context history purge is retained; only retrieval/evaluation is
subsampled. This is a descriptive robustness check, not a new tuning objective.
"""
from pathlib import Path
import ast
import json
import hashlib
import gc
SENSITIVITY_DRIVER=Path(__file__).resolve()
micro_source=SENSITIVITY_DRIVER.with_name('microcat_v6.py')
tree=ast.parse(micro_source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(micro_source)
exec(compile(tree,str(micro_source),'exec'),globals())
__file__=str(SENSITIVITY_DRIVER)
SENSITIVITY_CACHE=ROOT/'artifacts/cold-sensitivity-v5'
SENSITIVITY_CACHE.mkdir(exist_ok=True)
SENSITIVITY_FP=hashlib.sha256((V5_FP+hashlib.sha256(SENSITIVITY_DRIVER.read_bytes()).hexdigest()).encode()).hexdigest()[:16]


def run_sensitivity():
    random=np.random.default_rng(SEED+904)
    positions=np.sort(random.choice(len(development),400,replace=False))
    frame=development.iloc[positions]
    truth=labels_from_gold(gold,frame)
    all_truth=labels_from_gold(gold,development)
    other_truth=labels_from_gold(gold,control)
    other_positive=np.array(sorted({ITEM_IDS[i] for labels in other_truth for i in labels}),dtype=str)
    other_random=np.random.default_rng(SEED+812)
    other_cold=set(other_random.choice(other_positive,int(.9*len(other_positive)),replace=False))
    rows=[]
    for mode in ['unseen_text','held_context']:
        for fraction in [0.,.5,.9,1.]:
            path=SENSITIVITY_CACHE/f'{mode}_{fraction}_{SENSITIVITY_FP}.joblib'
            if USE_CACHE and path.exists():
                records,features,known,audit=joblib.load(path)
            else:
                fit,_,full_known=history_for_queries(history_all,development,all_truth,ITEM_IDS,mode,fraction,
                    SEED+801,excluded_texts=control.query_norm,excluded_items=other_cold)
                known=full_known[positions]
                history=V5History(fit)
                records=retrieve_semantic_features(frame,history,progress_every=400)
                features=[v5_features(q,r,history) for q,r in zip(frame.itertuples(index=False),records)]
                audit=history_coverage(frame,truth,fit,ITEM_IDS)
                save_cache((records,features,known,audit),path)
                del fit,history
            scores=baseline_v5(records,features)
            recalls=per_query_recall(top50(records,scores),truth)
            row={'mode':mode,'cold_fraction':fraction,'contexts':len(frame),'recall50':float(recalls.mean()),
                'pool_recall':float(per_query_recall([r[0] for r in records],truth).mean()),
                'actually_known_recall50':float(recalls[known].mean()) if known.any() else None,**audit}
            rows.append(row)
            print('COLD SENSITIVITY',json.dumps(row),flush=True)
            del records,features,scores
            gc.collect()
    pd.DataFrame(rows).to_csv(SENSITIVITY_CACHE/'recall_sensitivity.csv',index=False,lineterminator='\n')
    report={'fingerprint':SENSITIVITY_FP,'frozen_model':'v5','contexts':400,
        'positions':positions.tolist(),'same_contexts_across_settings':True,'full_history_purge_contexts':len(development),
        'new_models_fitted':False,'used_for_model_selection':False,'control_evaluated':False,
        'rows':rows,'answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation':'400 previously viewed development contexts; bootstrap uncertainty and test cold-item rate are not identified.'}
    assert report['answer_unchanged']
    (SENSITIVITY_CACHE/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    run_sensitivity()
