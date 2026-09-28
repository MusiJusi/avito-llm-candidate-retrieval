"""Test the trained cross-encoder against the fixed best v10 ensemble."""
from pathlib import Path
import ast
import gc
import json

CROSS_JOINT_DRIVER=Path(__file__).resolve()
source=CROSS_JOINT_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(CROSS_JOINT_DRIVER)
from activate_warm_history import activate
activate(globals())
from v10_validation_scores import install,compute
CE_CACHE=ROOT/'artifacts/cross-finetune-v10'


def values(stage,mode,ce_selection):
    frame,records,reference,known=compute(globals(),stage,mode)
    path=CE_CACHE/f'{stage}_{mode}_scores_{ce_selection["fingerprint"]}.joblib'
    score=joblib.load(path)
    assert len(score)==len(records)
    assert all(len(s)==len(r[0]) for s,r in zip(score,records))
    truth=labels_from_gold(gold,frame)
    output={0.:per_query_recall(top50(records,reference),truth)}
    for weight in [.025,.05,.1,.2]:
        output[weight]=per_query_recall(top50(records,blend_scores(score,reference,weight)),truth)
    del records,reference,score;gc.collect()
    return output,known


def compare():
    install(globals())
    ce_selection=json.loads((CE_CACHE/'selection.json').read_text())
    a,_=values('development','unseen_text',ce_selection)
    b,known=values('development','held_context',ce_selection)
    rows=[dict(weight=w,**matched_metrics(a[w],b[w],known)) for w in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(CE_CACHE/'joint_development.csv',index=False,lineterminator='\n')
    baseline=next(row for row in rows if row['weight']==0.)
    winner=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)&(table.held_macro>=baseline['held_macro']-.0005)].iloc[0].to_dict()
    report={'winner':winner,'baseline':baseline,'control_used_for_selection':False,
        'model':ce_selection['model'],'fingerprint':ce_selection['fingerprint'],
        'shortlist':'Frozen v9 + category top-200; candidates outside receive the fixed low score.',
        'upstream_quality_selection_sha256':sha256_file(V10_CACHE/'selection.json'),
        'upstream_warm_aux_selection_sha256':sha256_file(V10_CACHE/'warm_aux_selection.json'),
        'upstream_field_selection_sha256':sha256_file(ROOT/'artifacts/field-ranker-v10/selection.json'),
        'limitations':['Reused development/control; pilot trained on 6000 contexts only.']}
    (CE_CACHE/'joint_selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Cross encoder joint comparison',table.to_string(index=False),flush=True)
    # Pilot scores exist because its own development selection precedes control.
    if (CE_CACHE/f'control_unseen_text_scores_{ce_selection["fingerprint"]}.joblib').exists():
        c,_=values('control','unseen_text',ce_selection)
        (CE_CACHE/'joint_control.json').write_text(json.dumps({'baseline_recall50':float(c[0.].mean()),
            'selected_recall50':float(c[winner['weight']].mean()),'used_for_selection':False},indent=2),encoding='utf-8')


if __name__=='__main__':compare()
