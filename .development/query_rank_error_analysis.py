"""Analyze fixed E5 ranker on development; no tuning or hidden-test labels."""
from pathlib import Path
import ast
import json
import gc
DRIVER=Path(__file__).resolve()
source=DRIVER.with_name('query_encoder_rank.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())


def analyze_errors():
    selected=json.loads((QR_CACHE/'selection.json').read_text())['winner']
    assert selected['variant']=='query_ranker'
    model=joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    vectors=evaluation_query_features(development,'development')
    lookup={text:i for i,text in enumerate(sorted(set(development.query_norm)))}
    truth=labels_from_gold(gold,development)
    train_micros=set(history_all.item_microcat_id)
    query_rows=[];positive_rows=[]
    for mode in ['unseen_text','held_context']:
        records,features,known,audit=evaluation_features(development,mode,'development',control)
        matrix=[np.column_stack([x,query_features(vectors[lookup[text]],ids,raw)])
            for text,(ids,raw),x in zip(development.query_norm,records,features)]
        before=joblib.load(MICRO_CACHE/f'development_{mode}_v5scores_{MICRO_FP}.joblib')
        after=blend_scores(predict_scores(model,matrix,int(selected['trees'])),before,float(selected['weight']))
        old_predictions=top50(records,before);new_predictions=top50(records,after)
        old_recall=per_query_recall(old_predictions,truth);new_recall=per_query_recall(new_predictions,truth)
        for position,(query,(ids,raw),gold_ids) in enumerate(zip(development.itertuples(index=False),records,truth)):
            query_rows.append({'mode':mode,'context_key':query.context_key,'query_norm':query.query_norm,
                'v5_recall50':float(old_recall[position]),'query_recall50':float(new_recall[position]),
                'delta':float(new_recall[position]-old_recall[position]),'history_text_known':bool(known[position]),
                'positive_count':len(gold_ids),'pool_positive_count':len(set(ids)&set(gold_ids))})
            order=np.argsort(-after[position],kind='stable')
            ranks=np.empty(len(ids),np.int32);ranks[order]=np.arange(1,len(ids)+1)
            positions={int(item):i for i,item in enumerate(ids)}
            old_set=set(old_predictions[position]);new_set=set(new_predictions[position])
            for item in gold_ids:
                at=positions.get(int(item))
                positive_rows.append({'mode':mode,'context_key':query.context_key,'query_norm':query.query_norm,
                    'item_id':ITEM_IDS[item],'title':items.iloc[item].item_title_raw,
                    'same_location':int(query.search_location_id)==int(ITEM_LOCS[item]),
                    'center_available':int(query.search_location_id) in centers.index,
                    'microcat_seen_in_train':int(items.iloc[item].item_microcat_id) in train_micros,
                    'v5_hit50':item in old_set,'query_hit50':item in new_set,
                    'query_rank':int(ranks[at]) if at is not None else None,'in_pool':at is not None})
        del records,features,matrix,before,after
        gc.collect()
    queries_report=pd.DataFrame(query_rows);positives_report=pd.DataFrame(positive_rows)
    queries_report.to_csv(QR_CACHE/'development_query_deltas.csv',index=False,lineterminator='\n')
    positives_report.to_csv(QR_CACHE/'development_positive_ranks.csv',index=False,lineterminator='\n')
    result={}
    for mode,rows in positives_report.groupby('mode'):
        missing=rows[~rows.query_hit50]
        query_slice=queries_report[queries_report['mode']==mode]
        result[mode]={'positive_pairs':len(rows),'missed_pairs':len(missing),
            'outside_pool':int((~missing.in_pool).sum()),
            'ranks51_100':int(missing.query_rank.between(51,100).sum()),
            'ranks101_300':int(missing.query_rank.between(101,300).sum()),
            'ranks301_plus':int((missing.query_rank>300).sum()),
            'improved_contexts':int((query_slice.delta>0).sum()),'worsened_contexts':int((query_slice.delta<0).sum()),
            'same_location_pair_hit50':float(rows[rows.same_location].query_hit50.mean()),
            'different_location_pair_hit50':float(rows[~rows.same_location].query_hit50.mean())}
    names=V5_FEATURE_NAMES+QR_FEATURE_NAMES
    importance=model.booster_.feature_importance(importance_type='gain')
    pd.DataFrame({'feature':names,'gain':importance}).sort_values('gain',ascending=False).to_csv(QR_CACHE/'feature_gain.csv',index=False,lineterminator='\n')
    result['note']='Development already used for selection. Location slices are pair hit fractions, not macro query Recall. Gain is descriptive, not causal.'
    result['answer_unchanged']=sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH
    (QR_CACHE/'error_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('QUERY ERROR SUMMARY',json.dumps(result),flush=True)


if __name__=='__main__':analyze_errors()
