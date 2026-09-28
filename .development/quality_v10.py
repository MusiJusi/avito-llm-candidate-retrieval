"""Independent, bounded quality experiments on the frozen v9 candidate pools.

Category is an input supplied by the task, not an inferred test label. Start
with a soft compatibility prior: cross-category positives are retained. All
configuration selection uses development before inspecting reused control.
Frozen v7/v9 numerical producers and submitted files are never overwritten.
"""
from pathlib import Path
import ast
import argparse
import gc
import json
import time

DRIVER_V10 = Path(__file__).resolve()
source = DRIVER_V10.with_name('context_scale_v9.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n, ast.If)
    and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
    and n.test.left.id == '__name__')]
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())
__file__ = str(DRIVER_V10)
V9_FP = CONTEXT_FP
V9_CACHE = CONTEXT_CACHE
V10_CACHE = ROOT / 'artifacts/quality-v10'
V10_CACHE.mkdir(exist_ok=True)
ITEM_CATEGORIES = pd.to_numeric(items.item_category_id, errors='coerce').to_numpy()


def category_audit():
    """Describe the permitted category signal without benchmark labels."""
    pairs = pd.read_parquet(ROOT/'train.parquet',
        columns=['search_category', 'item_category_id', 'item_microcat_id'])
    same = pd.to_numeric(pairs.search_category, errors='coerce').eq(
        pd.to_numeric(pairs.item_category_id, errors='coerce'))
    summary = {'train_pairs':len(pairs), 'same_category_fraction':float(same.mean()),
        'train_search_categories':pairs.search_category.value_counts(dropna=False).to_dict(),
        'benchmark_search_categories':queries.search_category.value_counts(dropna=False).to_dict(),
        'corpus_item_categories':items.item_category_id.value_counts(dropna=False).to_dict(),
        'cross_category_positives':int((~same).sum()), 'hard_filter_used':False}
    (V10_CACHE/'category_audit.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print('Category audit',json.dumps(summary,default=str),flush=True)
    del pairs;gc.collect()


def v9_pool(stage, mode):
    frame = development if stage=='development' else control
    other = control if stage=='development' else development
    records, features, all_features = joblib.load(V9_CACHE/f'{stage}_{mode}_pools_{V9_FP}.joblib')
    cache = V10_CACHE/f'{stage}_{mode}_v9_scores.joblib'
    if cache.exists():
        scores = joblib.load(cache)
    else:
        baseline = baseline_v5(records,features)
        reference = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
        v7 = blend_scores(predict_scores(reference,[x[:,:51] for x in all_features],400),baseline,.75)
        model = joblib.load(V9_CACHE/f'evaluation_combined_{V9_FP}.joblib')
        scores = blend_scores(predict_scores(model,all_features,400),v7,.25)
        save_cache(scores,cache)
        del baseline,reference,v7,model
    history = evaluation_history(frame,mode,other)
    known = frame.query_norm.isin(history.query_norm).to_numpy()
    del history,features;gc.collect()
    return frame,records,all_features,scores,known


def compatibility_boost(frame, records, scores, strength):
    """A global soft prior in RRF score units; no test-ID-dependent rules."""
    if strength==0:return scores
    return [s+strength*(ITEM_CATEGORIES[ids]==float(q.search_category))
        for q,(ids,_),s in zip(frame.itertuples(index=False),records,scores)]


def matched_metrics(unseen, held, known):
    benchmark_known = queries.query_norm.isin(history_all.query_norm)
    a=matched_slice_recall(development,unseen,queries[~benchmark_known])
    b=matched_slice_recall(development,held,queries[benchmark_known],known)
    return dict(matched_recall50=(1-known_target)*a+known_target*b,
        unseen_macro=float(unseen.mean()),held_macro=float(held.mean()))


def category_experiment():
    category_audit()
    # Predetermined short grid. Incompatible categories are never deleted.
    strengths=[0.,.00005,.0001,.0002,.0004,.0008,.0016]
    values={};known=None;baseline_values={}
    for mode in ['unseen_text','held_context']:
        frame,records,features,scores,known=v9_pool('development',mode)
        truth=labels_from_gold(gold,frame)
        values[mode]={}
        for strength in strengths:
            prediction=top50(records,compatibility_boost(frame,records,scores,strength))
            values[mode][strength]=per_query_recall(prediction,truth)
        baseline_values[mode]=values[mode][0.]
        del records,features,scores;gc.collect()
    rows=[dict(category_boost=strength,**matched_metrics(values['unseen_text'][strength],
        values['held_context'][strength],known)) for strength in strengths]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(V10_CACHE/'category_development.csv',index=False,lineterminator='\n')
    baseline=next(row for row in rows if row['category_boost']==0.)
    assert abs(baseline['matched_recall50']-.9540428139856304)<1e-9,baseline
    eligible=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)
        &(table.held_macro>=baseline['held_macro']-.0005)]
    winner=eligible.iloc[0].to_dict()
    selection={'winner':winner,'baseline':baseline,'control_used_for_selection':False,
        'category_grid':strengths,'source_sha256':sha256_file(DRIVER_V10),
        'limitations':['Reused development and control; existing auxiliary-prior limitations remain.']}
    (V10_CACHE/'category_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Category development',table.to_string(index=False),flush=True)
    frame,records,features,scores,_=v9_pool('control','unseen_text')
    truth=labels_from_gold(gold,frame)
    a=per_query_recall(top50(records,scores),truth)
    b=per_query_recall(top50(records,compatibility_boost(frame,records,scores,winner['category_boost'])),truth)
    report={'v9_recall50':float(a.mean()),'selected_recall50':float(b.mean()),
        'improved':int((b>a).sum()),'worsened':int((b<a).sum()),'used_for_selection':False}
    (V10_CACHE/'category_control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Category control',json.dumps(report),flush=True)


if __name__=='__main__':
    category_experiment()
