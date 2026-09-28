"""Ablate item popularity under deliberate positive-item history erasure."""
from pathlib import Path
import ast
import gc
import json

NO_POP_DRIVER=Path(__file__).resolve()
source=NO_POP_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(NO_POP_DRIVER)
from activate_warm_history import activate
activate(globals())


def compare():
    global CONTEXT_VECTORS
    CONTEXT_VECTORS,_=contextual_vectors()
    data=load_quality_training('evaluation')
    model=fit_quality(data,'evaluation',['quality_without_pop'])['quality_without_pop']
    del data;gc.collect()
    category=json.loads((V10_CACHE/'category_selection.json').read_text())['winner']['category_boost']
    a,_=evaluate_quality('development','unseen_text',{'quality_without_pop':model},category)
    b,known=evaluate_quality('development','held_context',{'quality_without_pop':model},category)
    rows=[dict(variant=name,weight=weight,**matched_metrics(a[(name,weight)],b[(name,weight)],known)) for name,weight in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(V10_CACHE/'no_popularity_development.csv',index=False,lineterminator='\n')
    selection=json.loads((V10_CACHE/'selection.json').read_text());previous=dict(selection['winner'])
    eligible=table[(table.unseen_macro>=previous['unseen_macro']-.0005)&(table.held_macro>=previous['held_macro']-.0005)]
    best=eligible.iloc[0].to_dict() if len(eligible) else previous
    selection['no_popularity_ablation']=True
    selection['recipes']['quality_without_pop']=MODEL_RECIPES['quality_without_pop']
    if best['matched_recall50']>previous['matched_recall50']:
        selection['winner_before_no_popularity']=previous;selection['winner']=best
    (V10_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    c,_=evaluate_quality('control','unseen_text',{'quality_without_pop':model},category)
    chosen=table[table.variant=='quality_without_pop'].iloc[0]
    report={'selected_recall50':float(c[('quality_without_pop',float(chosen['weight']))].mean()),
        'winner':chosen.to_dict(),'primary_selection_updated':selection['winner']['variant']=='quality_without_pop',
        'used_for_selection':False}
    (V10_CACHE/'no_popularity_control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['primary_selection_updated']:
        old=json.loads((V10_CACHE/'control.json').read_text())
        (V10_CACHE/'control.json').write_text(json.dumps(dict(old,selected_recall50=report['selected_recall50'],
            before_no_popularity=old),indent=2),encoding='utf-8')
    print('No-popularity comparison',table.to_string(index=False),flush=True)


if __name__=='__main__':compare()
