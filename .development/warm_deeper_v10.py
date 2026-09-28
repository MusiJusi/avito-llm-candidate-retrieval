"""Capacity-matched warm-history ablation on the frozen mixed dataset."""
from pathlib import Path
import ast
import gc
import json

WARM_DEEP_DRIVER=Path(__file__).resolve()
source=WARM_DEEP_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(WARM_DEEP_DRIVER)
from activate_warm_history import activate
activate(globals())


def compare():
    global CONTEXT_VECTORS,QualityEvidence
    CONTEXT_VECTORS,_=contextual_vectors()
    data=mixed_training('evaluation')
    model=fit_quality(data,'evaluation',['warm_deeper'])['warm_deeper']
    del data;gc.collect()
    QualityEvidence=QualityEvidenceWithTrace
    selection=json.loads((V10_CACHE/'selection.json').read_text());previous=dict(selection['winner'])
    category=selection['category_boost']
    a,_=evaluate_quality('development','unseen_text',{'warm_deeper':model},category)
    b,known=evaluate_quality('development','held_context',{'warm_deeper':model},category)
    rows=[dict(variant=name,weight=weight,**matched_metrics(a[(name,weight)],b[(name,weight)],known)) for name,weight in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(V10_CACHE/'warm_deeper_development.csv',index=False,lineterminator='\n')
    eligible=table[(table.unseen_macro>=previous['unseen_macro']-.0005)&(table.held_macro>=previous['held_macro']-.0005)]
    best=eligible.iloc[0].to_dict() if len(eligible) else previous
    selection['recipes']['warm_deeper']=MODEL_RECIPES['warm_deeper']
    promoted=best['matched_recall50']>previous['matched_recall50']
    if promoted:
        selection['winner_before_warm_deeper']=previous;selection['winner']=best
        selection['warm_training_fingerprint']=WARM_FP
    (V10_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    chosen=best if promoted else table[table.variant=='warm_deeper'].iloc[0].to_dict()
    c,_=evaluate_quality('control','unseen_text',{'warm_deeper':model},category)
    score=float(c[('warm_deeper',float(chosen['weight']))].mean())
    report={'winner':chosen,'selected_recall50':score,'primary_selection_updated':promoted,'used_for_selection':False}
    (V10_CACHE/'warm_deeper_control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if promoted:
        old=json.loads((V10_CACHE/'control.json').read_text())
        (V10_CACHE/'control.json').write_text(json.dumps(dict(old,before_warm_deeper=old,selected_recall50=score),indent=2),encoding='utf-8')
    print('Warm capacity comparison',table.to_string(index=False),flush=True)


if __name__=='__main__':compare()
