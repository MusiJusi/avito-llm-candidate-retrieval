"""Validate and freeze the development-selected expanded-pool candidate.

The original v7 models and encoder remain fixed. This exports a separate answer,
and a self-contained notebook with the exact enlarged retrieval function. Final
quality must be checked on the platform; no benchmark labels are available.
"""
from pathlib import Path
import ast
import hashlib
import json
import gc

DRIVER = Path(__file__).resolve()
source = DRIVER.with_name('expanded_pool_rank_v8.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__ = str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__ = str(DRIVER)

def check_control():
    global LEARNED_LOOKUP
    selected = json.loads((EXPANDED_CACHE/'pilot_report.json').read_text())
    assert selected['after_matched_recall50']>selected['before_matched_recall50']
    install_expanded_retriever()
    vectors = evaluation_query_features(control,'control')
    LEARNED_LOOKUP = dict(zip(sorted(set(control.query_norm)),vectors))
    ranker = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    original_records,original_features,_,audit = evaluation_features(control,'unseen_text','control',development)
    original_base = baseline_v5(original_records,original_features)
    original_matrix = [np.column_stack([x,query_features(LEARNED_LOOKUP[text],ids,raw)])
        for text,(ids,raw),x in zip(control.query_norm,original_records,original_features)]
    truths = labels_from_gold(gold,control)
    before = per_query_recall(top50(original_records,blend_scores(
        predict_scores(ranker,original_matrix,400),original_base,.75)),truths)
    del original_records,original_features,original_base,original_matrix
    gc.collect()
    history_frame = evaluation_history(control,'unseen_text',development)
    history = V5History(history_frame)
    del history_frame
    records = retrieve_expanded_features(control,history,progress_every=200)
    features = [v5_features(query,record,history) for query,record in zip(control.itertuples(index=False),records)]
    baseline = baseline_v5(records,features)
    matrices = [np.column_stack([x,query_features(LEARNED_LOOKUP[text],ids,raw)])
        for text,(ids,raw),x in zip(control.query_norm,records,features)]
    after = per_query_recall(top50(records,blend_scores(predict_scores(ranker,matrices,400),baseline,.75)),truths)
    random = np.random.default_rng(SEED+1004)
    delta = after-before
    bootstrap = [float(random.choice(delta,len(delta),replace=True).mean()) for _ in range(3000)]
    report = {'v7_recall50':float(before.mean()),'expanded_recall50':float(after.mean()),
        'delta':float(delta.mean()),'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),
        'paired_bootstrap95':np.quantile(bootstrap,[.025,.975]).tolist(),'history_audit':audit,
        'used_for_configuration_selection':False,'previously_viewed_control':True,
        'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH}
    assert report['main_answer_unchanged'] and abs(before.mean()-.9533333333333334)<1e-10
    (EXPANDED_CACHE/'control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Expanded control',json.dumps(report),flush=True)

def export_expanded():
    global LEARNED_LOOKUP
    source = install_expanded_retriever()
    manifest = json.loads((ROOT/'artifacts/query-encoder-candidate/manifest.json').read_text())
    development_report = json.loads((EXPANDED_CACHE/'pilot_report.json').read_text())
    control_report = json.loads((EXPANDED_CACHE/'control.json').read_text())
    manifest['expanded_pool'] = {'per_channel':500,'development':development_report,'control':control_report}
    manifest['fingerprint'] = hashlib.sha256((source+manifest['final_ranker_sha256']).encode()).hexdigest()[:16]
    output_dir = ROOT/'experiments/results/expanded_pool_v8'
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest['answer_file'] = (output_dir/'answer.csv').relative_to(ROOT).as_posix()
    vectors = np.load(ROOT/manifest['query_vectors'],allow_pickle=False)
    LEARNED_LOOKUP = dict(zip(sorted(set(queries.query_norm)),vectors))
    history = V5History(history_all)
    records = retrieve_expanded_features(queries,history,progress_every=500)
    features = [v5_features(query,record,history) for query,record in zip(queries.itertuples(index=False),records)]
    baseline = baseline_v5(records,features,final=True)
    matrices = [np.column_stack([x,query_features(LEARNED_LOOKUP[text],ids,raw)])
        for text,(ids,raw),x in zip(queries.query_norm,records,features)]
    model = joblib.load(ROOT/manifest['final_ranker'])
    scores = blend_scores(predict_scores(model,matrices,400),baseline,.75)
    predictions = top50(records,scores)
    answer = pd.DataFrame({'query_id':queries.query_id.astype(str),
        'answer':[' '.join(ITEM_IDS[p]) for p in predictions]})
    allowed = set(ITEM_IDS)
    assert len(answer)==2452 and answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    for row in answer.itertuples(index=False):
        ids = row.answer.split(' ')
        assert len(row.query_id)==16 and len(ids)==len(set(ids))==50
        assert all(item in allowed and re.fullmatch('[0-9a-f]{16}',item) for item in ids)
    answer.to_csv(ROOT/manifest['answer_file'],index=False,encoding='utf-8',lineterminator='\n')
    manifest['answer_sha256'] = sha256_file(ROOT/manifest['answer_file'])
    assert sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH
    (EXPANDED_CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    notebook = json.loads((ROOT/'Avito.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]['source'] = ['# Кандидат v8: дополнительный поиск дообученной E5\n',
        '\nRun All создаёт experiments/results/expanded_pool_v8/answer.csv.\n',
        'Платформа: ещё не проверено. Основной v7 ответ сохранён отдельно.\n']
    cell = {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
        'source':('LEARNED_LOOKUP = {}\n\n'+source+'\n').splitlines(True),
        'id':hashlib.sha256(source.encode()).hexdigest()[:12]}
    notebook['cells'].insert(-1,cell)
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        text = ''.join(cell['source'])
        text = text.replace('ROOT = Path.cwd()',
            "ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'train.parquet').exists()), Path.cwd())")
        if text.startswith('FROZEN_QUERY_MANIFEST = '):
            text = 'FROZEN_QUERY_MANIFEST = '+repr(manifest)+'\n'+text[text.index('\n')+1:]
        if text.startswith('def export_frozen_query_candidate():'):
            text = text.replace('def export_frozen_query_candidate():',
                'def export_frozen_query_candidate():\n    global LEARNED_LOOKUP')
            text = text.replace("pool_path=V5_CACHE/f'benchmark_{V5_FP}.joblib'",
                "LEARNED_LOOKUP = {text:vectors[row] for text,row in lookup.items()}\n    pool_path=V5_CACHE/f'expanded_benchmark_{FROZEN_QUERY_MANIFEST[\"fingerprint\"]}.joblib'")
            text = text.replace('records=retrieve_semantic_features(queries,history,progress_every=500)',
                'records=retrieve_expanded_features(queries,history,progress_every=500)')
        cell['source'] = text.splitlines(True)
        cell['outputs']=[];cell['execution_count']=None
        ast.parse(text)
    notebook['metadata'].pop('platform_recall50',None)
    (ROOT/'experiments/Avito_expanded_pool_v8_candidate.ipynb').write_text(json.dumps(notebook,
        ensure_ascii=False,indent=1),encoding='utf-8')
    print('Expanded candidate exported',manifest['answer_file'],manifest['answer_sha256'],flush=True)

if __name__=='__main__':
    if not (EXPANDED_CACHE/'control.json').exists():check_control()
    export_expanded()
