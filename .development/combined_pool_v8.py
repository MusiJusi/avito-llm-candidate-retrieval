"""Compare the independently selected ranker on enlarged learned-E5 pools.

Configuration selection uses development only. A reused control is reported
afterwards; it is not an independent final test. No gold is injected into pools.
"""
from pathlib import Path
import ast
import json
import gc

driver = Path(__file__).resolve()
source = driver.with_name('expanded_pool_rank_v8.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n, ast.If)
    and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
    and n.test.left.id == '__name__')]
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())
__file__ = str(driver)
CACHE = ROOT/'artifacts/combined-pool-v8'
CACHE.mkdir(exist_ok=True)

def run():
    global LEARNED_LOOKUP
    install_expanded_retriever()
    selection = json.loads((ROOT/'artifacts/ranker-v8/selection.json').read_text())
    winner = selection['winner']
    model = joblib.load(ROOT/f'artifacts/ranker-v8/evaluation_{winner["model"]}_{selection["fingerprint"]}.joblib')
    reference = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    values = {}; saved_known = None; statistics = []
    for mode in ['unseen_text', 'held_context']:
        vectors = evaluation_query_features(development, 'development')
        LEARNED_LOOKUP = dict(zip(sorted(set(development.query_norm)), vectors))
        history_frame = evaluation_history(development, mode, control)
        known = development.query_norm.isin(history_frame.query_norm).to_numpy()
        history = V5History(history_frame)
        del history_frame
        records = retrieve_expanded_features(development, history, progress_every=1000)
        features = [v5_features(q, r, history) for q,r in zip(development.itertuples(index=False),records)]
        del history
        baseline = baseline_v5(records, features)
        matrices = [np.column_stack([x,query_features(LEARNED_LOOKUP[text],ids,raw)])
            for text,(ids,raw),x in zip(development.query_norm,records,features)]
        v7scores = blend_scores(predict_scores(reference, matrices, 400), baseline, .75)
        strong = blend_scores(predict_scores(model, matrices, int(winner['trees'])), baseline, float(winner['weight']))
        truth = labels_from_gold(gold, development)
        # A small prespecified ensemble grid can retain complementary mistakes.
        for weight in [0., .25, .5, .75, 1.]:
            scores = blend_scores(strong, v7scores, weight)
            values[(mode,weight)] = per_query_recall(top50(records,scores),truth)
        if mode=='held_context': saved_known=known
        frame = pd.DataFrame({'context_key':development.context_key})
        for weight in [0., .25, .5, .75, 1.]:frame[str(weight)] = values[(mode,weight)]
        frame.to_csv(CACHE/f'{mode}_queries.csv',index=False)
        del records,features,matrices,baseline,v7scores,strong
        gc.collect()
    query_known = queries.query_norm.isin(history_all.query_norm)
    rows=[]
    for weight in [0., .25, .5, .75, 1.]:
        a=matched_slice_recall(development,values[('unseen_text',weight)],queries[~query_known])
        b=matched_slice_recall(development,values[('held_context',weight)],queries[query_known],saved_known)
        rows.append({'strong_ranker_weight':weight,'matched_recall50':(1-known_target)*a+known_target*b,
            'unseen_macro':float(values[('unseen_text',weight)].mean()),
            'held_macro':float(values[('held_context',weight)].mean())})
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(CACHE/'development.csv',index=False)
    selected=table.iloc[0].to_dict()
    report={'winner':selected,'ranker_selection':selection,'control_used_for_selection':False,
        'baseline_v7_matched':.9501708535101647,'expanded_v7_matched':rows[0]['matched_recall50']}
    (CACHE/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Combined development',json.dumps(report),flush=True)
    # Freeze the development choice before running the previously viewed control.
    vectors=evaluation_query_features(control,'control')
    LEARNED_LOOKUP=dict(zip(sorted(set(control.query_norm)),vectors))
    history_frame=evaluation_history(control,'unseen_text',development)
    history=V5History(history_frame);del history_frame
    records=retrieve_expanded_features(control,history,progress_every=300)
    features=[v5_features(q,r,history) for q,r in zip(control.itertuples(index=False),records)]
    baseline=baseline_v5(records,features)
    matrices=[np.column_stack([x,query_features(LEARNED_LOOKUP[text],ids,raw)])
        for text,(ids,raw),x in zip(control.query_norm,records,features)]
    v7scores=blend_scores(predict_scores(reference,matrices,400),baseline,.75)
    strong=blend_scores(predict_scores(model,matrices,int(winner['trees'])),baseline,float(winner['weight']))
    scores=blend_scores(strong,v7scores,selected['strong_ranker_weight'])
    result=per_query_recall(top50(records,scores),labels_from_gold(gold,control))
    report={'combined_recall50':float(result.mean()),'v7_recall50':.9533333333333334,
        'previously_viewed_control':True,'used_for_selection':False,
        'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH}
    assert report['main_answer_unchanged']
    (CACHE/'control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Combined control',json.dumps(report),flush=True)

if __name__=='__main__':run()
