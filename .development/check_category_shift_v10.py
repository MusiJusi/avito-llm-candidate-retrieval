"""Counterfactual unknown-category stress check before benchmark inference.

The development labels mostly belong to category 114. Making the search category
unspecified broadens eligibility, so these positives remain valid. This is an
explicit stress proxy, not additional independent gold for category-0 queries.
"""
from pathlib import Path
import ast
import gc
import json

SHIFT_DRIVER=Path(__file__).resolve()
source=SHIFT_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(SHIFT_DRIVER)
from activate_warm_history import activate
activate(globals())


def stress():
    global QUALITY_BANK,CONTEXT_VECTORS
    CONTEXT_VECTORS,_=contextual_vectors()
    selection=json.loads((V10_CACHE/'selection.json').read_text());winner=selection['winner']
    name=winner['variant']
    model=joblib.load(V10_CACHE/f'evaluation_{name}_{V10_FP}.joblib') if name in MODEL_RECIPES else None
    results={};known=None
    for mode in ['unseen_text','held_context']:
        frame,records,base,reference,known=v9_pool('development',mode)
        history=evaluation_history(frame,mode,control)
        QUALITY_BANK=(QualityEvidenceWithTrace if name.startswith('warm_') else QualityEvidence)(history)
        del history;gc.collect()
        features=extra_matrices(frame,records,base)
        truth=labels_from_gold(gold,frame)
        results[mode]={'v9_without_category_prior':per_query_recall(top50(records,reference),truth)}
        if model is not None:
            recipe=MODEL_RECIPES[name]
            for policy,value in [('unspecified_as_mismatch',0.),('unspecified_as_compatible',1.)]:
                for x in features:x[:,70]=value
                score=predict_scores(model,[x[:,recipe['columns']] for x in features],recipe['trees'])
                score=blend_scores(score,reference,winner['weight'])
                results[mode][policy]=per_query_recall(top50(records,score),truth)
                del score;gc.collect()
            # Report actual validation behavior after the semantic correction.
            for q,(ids,_),x in zip(frame.itertuples(index=False),records,features):
                x[:,70]=1. if float(q.search_category)==0 else ITEM_CATEGORIES[ids]==float(q.search_category)
            actual_reference=compatibility_boost(frame,records,reference,selection['category_boost'])
            score=predict_scores(model,[x[:,recipe['columns']] for x in features],recipe['trees'])
            score=blend_scores(score,actual_reference,winner['weight'])
            results[mode]['actual_queries_wildcard_policy']=per_query_recall(top50(records,score),truth)
            del score,actual_reference;gc.collect()
        del records,base,features,reference;gc.collect()
    rows=[dict(policy=policy,**matched_metrics(results['unseen_text'][policy],results['held_context'][policy],known))
        for policy in results['unseen_text']]
    report={'results':rows,'benchmark_unspecified_category_fraction':float(queries.search_category.eq(0).mean()),
        'actual_development_unspecified_queries':int(development.search_category.eq(0).sum()),
        'weights_fixed':True,'source_sha256':sha256_file(SHIFT_DRIVER),
        'policy':'Category 0 means no category constraint; treat compatibility as true.',
        'limitation':'Counterfactual stress proxy, not an independent labeled sample of category-0 searches.'}
    (V10_CACHE/'category_shift_stress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    actual=next((row for row in rows if row['policy']=='actual_queries_wildcard_policy'),None)
    if actual:
        selection['unknown_category_policy']='compatible, no category constraint'
        selection['winner_before_category_policy_correction']=dict(selection['winner'])
        selection['winner'].update({key:actual[key] for key in ['matched_recall50','unseen_macro','held_macro']})
        (V10_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Unknown category stress',pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':stress()
