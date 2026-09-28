"""Describe remaining misses using development gold only, never test labels."""
from pathlib import Path
import ast
import gc
import json

ERROR_DRIVER=Path(__file__).resolve()
source=ERROR_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(ERROR_DRIVER)
from activate_warm_history import activate
activate(globals())
from v10_validation_scores import install,compute


def analyze():
    install(globals());rows=[];counts=[]
    ce_cache=ROOT/'artifacts/cross-finetune-v10'
    joint_path=ce_cache/'joint_selection.json'
    joint=json.loads(joint_path.read_text()) if joint_path.exists() else None
    for mode in ['unseen_text','held_context']:
        frame,records,scores,known=compute(globals(),'development',mode)
        if joint and joint['winner']['weight']>0:
            ce=joblib.load(ce_cache/f'development_{mode}_scores_{joint["fingerprint"]}.joblib')
            scores=blend_scores(ce,scores,joint['winner']['weight']);del ce
        truth=labels_from_gold(gold,frame);predictions=top50(records,scores)
        history=evaluation_history(frame,mode,control)
        seen=set(history.item_id);exact=set(zip(history.query_norm,history.item_id))
        del history;gc.collect()
        retrieval_miss=ranking_miss=queries_with_miss=0
        for q,(ids,raw),score,target,prediction in zip(frame.itertuples(index=False),records,scores,truth,predictions):
            missing=set(target)-set(prediction)
            if missing:queries_with_miss+=1
            order=stable_topk(score,len(score));ranks={int(ids[p]):rank+1 for rank,p in enumerate(order)}
            for item in missing:
                present=item in ranks
                retrieval_miss+=not present;ranking_miss+=present
                item_row=items.iloc[item]
                rows.append({'mode':mode,'query':q.query_norm,'filters':q.search_infm_params_text,
                    'miss_type':'ranking' if present else 'retrieval','positive_rank':ranks.get(item),
                    'positive_title':item_row.item_title_raw,
                    'same_location':int(q.search_location_id)==int(item_row.item_location_id),
                    'item_seen_in_permitted_history':str(item_row.item_id) in seen,
                    'query_item_seen_in_permitted_history':(q.query_norm,str(item_row.item_id)) in exact,
                    'positive_rating':item_row.item_rating,'positive_reviews':item_row.item_rating_reviews_count})
        counts.append({'mode':mode,'queries':len(frame),'queries_with_missed_positive':queries_with_miss,
            'retrieval_missed_positive_items':retrieval_miss,'ranking_missed_positive_items':ranking_miss})
        del frame,records,scores,predictions,seen,exact;gc.collect()
    table=pd.DataFrame(rows)
    table.to_csv(V10_CACHE/'development_misses.csv',index=False,lineterminator='\n')
    summary={'counts':counts,'gold_source':'development derived from train only; no benchmark labels',
        'unselected_items_not_proven_irrelevant':True,'cross_validation_precision':'Cached CUDA BF16 pilot if selected'}
    selected=json.loads((V10_CACHE/'selection.json').read_text())['winner']['variant']
    if selected in MODEL_RECIPES:
        model=joblib.load(V10_CACHE/f'evaluation_{selected}_{V10_FP}.joblib')
        names=NEW_FEATURE_NAMES+(TRACE_NAMES if selected.startswith('warm_') else [])
        columns=MODEL_RECIPES[selected]['columns']
        gains=model.booster_.feature_importance(importance_type='gain')
        best=np.argsort(-gains)[:12]
        summary['primary_model_top_features_by_gain']=[{
            'feature':names[columns[i]],'gain_share':float(gains[i]/max(gains.sum(),1e-12))} for i in best]
        summary['importance_is_descriptive_not_causal']=True
    if len(table):summary['ranking_misses_same_location_fraction']=float(table.query('miss_type == "ranking"').same_location.mean())
    (V10_CACHE/'error_analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Remaining quality errors',json.dumps(summary),flush=True)


if __name__=='__main__':analyze()
