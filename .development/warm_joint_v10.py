"""Check whether different-history models complement the fixed strong ensemble."""
from pathlib import Path
import ast
import gc
import json

WARM_JOINT_DRIVER=Path(__file__).resolve()
source=WARM_JOINT_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(WARM_JOINT_DRIVER)
from activate_warm_history import activate
activate(globals())
from v10_validation_scores import install,compute


def values(stage,mode,models):
    frame,records,reference,known,features=compute(globals(),stage,mode,return_warm_features=True)
    truth=labels_from_gold(gold,frame)
    result={('baseline',0.):per_query_recall(top50(records,reference),truth)}
    for name,model in models.items():
        prediction=predict_scores(model,features,MODEL_RECIPES[name]['trees'])
        for weight in [.15,.3,.5]:
            result[(name,weight)]=per_query_recall(top50(records,blend_scores(prediction,reference,weight)),truth)
        del prediction
    del records,reference,features;gc.collect()
    return result,known


def compare():
    # An existing result is immutable for this completed upstream ensemble.
    path=V10_CACHE/'warm_aux_selection.json'
    if path.exists():return
    install(globals())
    models={name:joblib.load(V10_CACHE/f'evaluation_{name}_{V10_FP}.joblib') for name in ['warm_mixed','warm_deeper']}
    a,_=values('development','unseen_text',models)
    b,known=values('development','held_context',models)
    rows=[dict(variant=name,weight=weight,**matched_metrics(a[(name,weight)],b[(name,weight)],known)) for name,weight in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(V10_CACHE/'warm_aux_development.csv',index=False,lineterminator='\n')
    baseline=next(row for row in rows if row['variant']=='baseline')
    winner=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)&(table.held_macro>=baseline['held_macro']-.0005)].iloc[0].to_dict()
    report={'winner':winner,'baseline':baseline,'control_used_for_selection':False,
        'upstream_quality_sha256':sha256_file(V10_CACHE/'selection.json'),
        'upstream_field_sha256':sha256_file(ROOT/'artifacts/field-ranker-v10/selection.json'),
        'fingerprint':V10_FP,'warm_training_fingerprint':WARM_FP,'source_sha256':sha256_file(WARM_JOINT_DRIVER),
        'limitation':'Reused development/control; fixed cold evaluation and fixed auxiliary encoders.'}
    # compute() must not apply this new auxiliary before its own control compare.
    (V10_CACHE/'warm_aux_choice_before_control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Warm-history joint comparison',table.to_string(index=False),flush=True)
    chosen={winner['variant']:models[winner['variant']]} if winner['variant'] in models else {}
    c,_=values('control','unseen_text',chosen)
    baseline_control=float(c[('baseline',0.)].mean())
    original_control=float(c[(winner['variant'],winner['weight'])].mean())
    # A sub-one-hit development gain is too fragile if control drops. Record
    # this control veto explicitly; the reported control is no longer an
    # independent estimate of the final choice.
    if original_control<baseline_control and winner['matched_recall50']-baseline['matched_recall50']<1/3400:
        report['before_control_veto']=winner
        report['winner']=baseline
        report['control_veto_applied']=True
        report['control_used_for_selection']=True
        report['source_sha256']=sha256_file(WARM_JOINT_DRIVER)
    (V10_CACHE/'warm_aux_control.json').write_text(json.dumps({
        'baseline_recall50':baseline_control,
        'original_selected_recall50':original_control,
        'selected_recall50':baseline_control if report.get('control_veto_applied') else original_control,
        'used_for_selection':bool(report.get('control_veto_applied'))},indent=2),encoding='utf-8')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':compare()
