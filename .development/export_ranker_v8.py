"""Refit the development-selected ranker and preserve an isolated candidate.

The confirmed primary submission stays intact. A candidate notebook loads exactly
the same fixed encoder/vectors and uses the selected new ranker. It is promoted
only after reproduction and independent CSV validation.
"""
from pathlib import Path
import ast
import hashlib
import json
import gc

DRIVER = Path(__file__).resolve()
source = DRIVER.with_name('ranker_v8.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__ = str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__ = str(DRIVER)

def export_candidate():
    selection = json.loads((V8_CACHE/'selection.json').read_text())
    winner = selection['winner']
    assert winner['model']!='v7','Development did not select an improved ranker'
    assert winner['matched_recall50']>selection['baseline']['matched_recall50']
    configuration = dict(selection['configuration'])
    configuration['n_estimators'] = int(winner['trees'])
    data = joblib.load(V5_CACHE/f'final_mined_{V5_FP}.joblib')
    original_manifest = json.loads((ROOT/'artifacts/query-encoder-candidate/manifest.json').read_text())
    final_fp = original_manifest['fingerprint']
    extra = joblib.load(ROOT/f'artifacts/query-encoder-candidate/oof_features_{final_fp}.joblib')
    model = fitted_model(data,extra,winner['model'],configuration,stage='final')
    del data,extra
    gc.collect()
    ranker_path = V8_CACHE/f'final_{winner["model"]}_{V8_FP}.joblib'
    manifest = dict(original_manifest)
    manifest['final_ranker'] = ranker_path.relative_to(ROOT).as_posix()
    manifest['final_ranker_sha256'] = sha256_file(ranker_path)
    manifest['selection'] = {'winner':{'variant':'query_ranker','trees':int(winner['trees']),
        'weight':float(winner['weight'])},'development':selection,'configuration':configuration}
    manifest['control'] = json.loads((V8_CACHE/'control.json').read_text())
    manifest['fingerprint'] = V8_FP
    output_dir = ROOT/'experiments/results/ranker_v8'
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest['answer_file'] = (output_dir/'answer.csv').relative_to(ROOT).as_posix()
    # Score the frozen benchmark pools with the full-train encoder from confirmed v7.
    records,features = joblib.load(V5_CACHE/f'benchmark_{V5_FP}.joblib')
    vectors = np.load(ROOT/original_manifest['query_vectors'],allow_pickle=False)
    lookup = {text:i for i,text in enumerate(sorted(set(queries.query_norm)))}
    matrices = [np.column_stack([x,query_features(vectors[lookup[text]],ids,raw)])
        for text,(ids,raw),x in zip(queries.query_norm,records,features)]
    baseline = baseline_v5(records,features,final=True)
    predictions = top50(records,blend_scores(predict_scores(model,matrices,int(winner['trees'])),
        baseline,float(winner['weight'])))
    answer = pd.DataFrame({'query_id':queries.query_id.astype(str),
        'answer':[' '.join(ITEM_IDS[p]) for p in predictions]})
    assert len(answer)==2452 and answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    allowed = set(ITEM_IDS)
    for row in answer.itertuples(index=False):
        ids = row.answer.split(' ')
        assert len(row.query_id)==16 and len(ids)==len(set(ids))==50
        assert all(item in allowed and re.fullmatch('[0-9a-f]{16}',item) for item in ids)
    output = ROOT/manifest['answer_file']
    answer.to_csv(output,index=False,encoding='utf-8',lineterminator='\n')
    manifest['answer_sha256'] = sha256_file(output)
    manifest['main_answer_unchanged'] = sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH
    assert manifest['main_answer_unchanged']
    (V8_CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    notebook = json.loads((ROOT/'Avito.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]['source'] = ['# Кандидат v8: усиленный ранкер\n',
        '\nRestart Kernel → Run All создаёт experiments/results/ranker_v8/answer.csv.\n',
        'Параметры выбраны на development; метрика платформы пока неизвестна.\n']
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            value = ''.join(cell['source'])
            value = value.replace('ROOT = Path.cwd()',
                "ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'train.parquet').exists()), Path.cwd())")
            if value.startswith('FROZEN_QUERY_MANIFEST = '):
                value = 'FROZEN_QUERY_MANIFEST = '+repr(manifest)+'\n'+value[value.index('\n')+1:]
            cell['source'] = value.splitlines(True)
            cell['outputs'] = [];cell['execution_count']=None
            ast.parse(value)
    notebook['metadata'].pop('platform_recall50',None)
    (ROOT/'experiments/Avito_ranker_v8_candidate.ipynb').write_text(json.dumps(notebook,
        ensure_ascii=False,indent=1),encoding='utf-8')
    print('Validated v8 candidate',manifest['answer_file'],manifest['answer_sha256'],flush=True)

if __name__=='__main__':
    export_candidate()
