"""Describe errors of the already selected model, without selecting on control."""
from pathlib import Path
import ast
import json
import gc

ANALYSIS_DRIVER = Path(__file__).resolve()
source = ANALYSIS_DRIVER.with_name('context_scale_v9.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')]
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())
__file__ = str(ANALYSIS_DRIVER)

def analyze():
    selection = json.loads((CONTEXT_CACHE/'selection.json').read_text())
    winner = selection['winner']
    if winner['variant'] not in FEATURE_SETS:
        print('No new selected ranker to analyze.');return
    frame = control
    path = CONTEXT_CACHE/f'control_unseen_text_pools_{CONTEXT_FP}.joblib'
    records, base_features, features = joblib.load(path)
    baseline = baseline_v5(records, base_features)
    incumbent = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    old_scores = blend_scores(predict_scores(incumbent,[x[:,:51] for x in features],400),baseline,.75)
    model = joblib.load(CONTEXT_CACHE/f'evaluation_{winner["variant"]}_{CONTEXT_FP}.joblib')
    columns = FEATURE_SETS[winner['variant']]
    scores = predict_scores(model,[np.ascontiguousarray(x[:,columns]) for x in features],400)
    selected_scores = blend_scores(scores,old_scores,winner['weight'])
    truths = labels_from_gold(gold,frame)
    old = per_query_recall(top50(records,old_scores),truths)
    new = per_query_recall(top50(records,selected_scores),truths)
    assert abs(new.mean()-json.loads((CONTEXT_CACHE/'control.json').read_text())['selected_recall50'])<1e-12
    rows = frame[['search_query','search_infm_params_text','search_location_id']].copy()
    rows['incumbent_recall50']=old;rows['selected_recall50']=new;rows['delta']=new-old
    rows['geography_source']=[float(x[0,52]) for x in features]
    rows['geography_confidence']=[float(x[0,56]) for x in features]
    rows['positives']=[len(t) for t in truths]
    rows.to_csv(CONTEXT_CACHE/'control_errors.csv',index=False,lineterminator='\n')
    report={'model_already_selected':True,'used_for_model_selection':False,
        'queries':len(frame),'improved':int((new>old).sum()),'worsened':int((new<old).sum()),
        'unchanged':int((new==old).sum()),'incumbent_recall50':float(old.mean()),
        'selected_recall50':float(new.mean()),'lost_examples':[], 'gained_examples':[]}
    for key,mask in [('lost_examples',new<old),('gained_examples',new>old)]:
        report[key]=rows[mask].head(8).to_dict('records')
    (CONTEXT_CACHE/'error_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':analyze()
