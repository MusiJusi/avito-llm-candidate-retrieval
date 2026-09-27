"""Complete the training-size ablation on the same 12,000-context OOF data.

Only the incumbent 51 feature columns are trained here. Selection compares
development results before inspecting this model on the previously viewed
control; it never alters training labels or history isolation.
"""
from pathlib import Path
import ast
import json
import gc

BASE_SCALE_DRIVER=Path(__file__).resolve()
source=BASE_SCALE_DRIVER.with_name('context_scale_v9.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[node for node in tree.body if not (
    isinstance(node,ast.If) and isinstance(node.test,ast.Compare)
    and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(BASE_SCALE_DRIVER)
FEATURE_SETS=ALL_FEATURE_SETS

def compare_base():
    data=joblib.load(CONTEXT_CACHE/f'evaluation_training_{CONTEXT_FP}.joblib',mmap_mode='r')
    models=fit_models(data,'evaluation',['base'])
    del data;gc.collect()
    a,_,_,_=evaluate(development,'unseen_text','development',models,control)
    b,known,_,_=evaluate(development,'held_context','development',models,control)
    query_known=queries.query_norm.isin(history_all.query_norm)
    rows=[]
    for key in a:
        if not isinstance(key,tuple):continue
        score_a=matched_slice_recall(development,a[key],queries[~query_known])
        score_b=matched_slice_recall(development,b[key],queries[query_known],known)
        rows.append(dict(variant=key[0],weight=key[1],matched_recall50=(1-known_target)*score_a+known_target*score_b,
            unseen_macro=float(a[key].mean()),held_macro=float(b[key].mean())))
    path=CONTEXT_CACHE/'development.csv'
    old=pd.read_csv(path)
    table=pd.concat([old[old.variant!='base'],pd.DataFrame(rows)],ignore_index=True).sort_values('matched_recall50',ascending=False)
    table.to_csv(path,index=False,lineterminator='\n')
    selection=json.loads((CONTEXT_CACHE/'selection.json').read_text())
    previous=selection['winner']
    winner=table[table.unseen_macro>=.9478676470588235-.002].iloc[0].to_dict()
    selection['winner']=winner;selection['completed_base_ablation']=True
    (CONTEXT_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Expanded base development comparison',json.dumps(winner),flush=True)
    if winner['variant']=='base' and previous['variant']!='base':
        value,_,audit,pool=evaluate(control,'unseen_text','control',models,development)
        result=dict(selected_recall50=float(value[('base',winner['weight'])].mean()),v8_recall50=.9533333333333334,
            pool_recall=pool,history_audit=audit,previously_viewed_control=True,used_for_selection=False)
        (CONTEXT_CACHE/'control.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Selected base control',json.dumps(result),flush=True)

if __name__=='__main__':compare_base()
