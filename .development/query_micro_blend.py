"""Check whether microcat and learned E5 correct complementary dev errors.

All model settings are already fixed. Five bounded RRF weights are selected on
development; only the selected combination is checked on previously viewed control.
This diagnostic does not change any final answer or train new components.
"""
from pathlib import Path
import ast
import hashlib
import json
import gc
DRIVER=Path(__file__).resolve()
source=DRIVER.with_name('query_encoder_rank.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
RANK_FP=hashlib.sha256((MICRO_FP+hashlib.sha256(DRIVER.with_name('microcat_rank_v6.py').read_bytes()).hexdigest()).encode()).hexdigest()[:16]
BLEND_CACHE=ROOT/'artifacts/query-micro-blend'
BLEND_CACHE.mkdir(exist_ok=True)


def score_both(frame,mode,stage):
    other=control if stage=='development' else development
    records,base,known,audit=evaluation_features(frame,mode,stage,other)
    original=joblib.load(MICRO_CACHE/f'{stage}_{mode}_v5scores_{MICRO_FP}.joblib') if stage=='development' else baseline_v5(records,base)
    qp=json.loads((QR_CACHE/'selection.json').read_text())['winner']
    mp=json.loads((MICRO_CACHE/'ranker_selection.json').read_text())['winner']
    assert qp['variant']=='query_ranker' and mp['variant']=='both'
    vectors=evaluation_query_features(frame,stage)
    lookup={text:i for i,text in enumerate(sorted(set(frame.query_norm)))}
    matrix=[np.column_stack([x,query_features(vectors[lookup[text]],ids,raw)])
        for text,(ids,raw),x in zip(frame.query_norm,records,base)]
    qm=joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    query=blend_scores(predict_scores(qm,matrix,int(qp['trees'])),original,float(qp['weight']))
    del matrix,qm
    probabilities=joblib.load(MICRO_CACHE/f'{stage}_{mode}_probabilities_{MICRO_FP}.joblib')
    classes=np.sort(history_all.item_microcat_id.unique())
    micro=[]
    for position,((ids,_),x) in enumerate(zip(records,base)):
        added=[micro_candidate_features(probabilities[k]['probabilities'][position],items.item_microcat_id.to_numpy()[ids],classes,probabilities[k]['supported']) for k in ['nb','mlp']]
        micro.append(np.column_stack([x,*added]))
    mm=joblib.load(MICRO_CACHE/f'evaluation_rank_both_{RANK_FP}.joblib')
    micro=blend_scores(predict_scores(mm,micro,int(mp['trees'])),original,float(mp['weight']))
    return records,query,micro,known


def check_blend():
    a=score_both(development,'unseen_text','development')
    b=score_both(development,'held_context','development')
    rows=[]
    for weight in [0.,.1,.2,.3,.5]:
        metrics=development_metrics(top50(a[0],blend_scores(a[2],a[1],weight)),
            top50(b[0],blend_scores(b[2],b[1],weight)),b[3])
        rows.append({'micro_weight':weight,**metrics})
    table=pd.DataFrame(rows).sort_values(['matched_recall50','micro_weight'],ascending=[False,True])
    table.to_csv(BLEND_CACHE/'ablation.csv',index=False,lineterminator='\n')
    baseline=table[table.micro_weight==0].iloc[0].to_dict()
    winner=table[table.unseen_macro_recall50>=baseline['unseen_macro_recall50']-.002].iloc[0].to_dict()
    report={'winner':winner,'baseline':baseline,'control_used_for_selection':False,
        'query_rank_fingerprint':QR_FP,'micro_rank_fingerprint':RANK_FP,'answer_unchanged':True}
    del a,b;gc.collect()
    if winner['micro_weight']:
        records,query,micro,_=score_both(control,'unseen_text','control')
        truth=labels_from_gold(gold,control)
        before=per_query_recall(top50(records,query),truth)
        after=per_query_recall(top50(records,blend_scores(micro,query,float(winner['micro_weight']))),truth)
        report['control']={'query_only':float(before.mean()),'blend':float(after.mean()),
            'improved':int((after>before).sum()),'worsened':int((after<before).sum()),
            'limitation':'Previously viewed control; frozen v4 auxiliary-history limitation.'}
    else:report['control']=None
    assert sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH
    (BLEND_CACHE/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('QUERY MICRO BLEND',json.dumps(report),flush=True)


if __name__=='__main__':check_blend()
