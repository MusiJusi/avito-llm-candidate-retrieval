"""Freeze the development-selected combination without replacing confirmed v7."""
from pathlib import Path
import ast
import hashlib
import json

driver = Path(__file__).resolve()
source = driver.with_name('expanded_pool_rank_v8.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n,ast.If)
    and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name)
    and n.test.left.id=='__name__')]
__file__ = str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__ = str(driver)

def export():
    global LEARNED_LOOKUP
    install_expanded_retriever()
    cache = ROOT/'artifacts/combined-pool-v8'
    selection = json.loads((cache/'selection.json').read_text())
    weight = float(selection['winner']['strong_ranker_weight'])
    old = json.loads((EXPANDED_CACHE/'manifest.json').read_text())
    strong = json.loads((ROOT/'artifacts/ranker-v8/manifest.json').read_text())
    manifest = dict(old if weight==0 else strong)
    manifest['expanded_pool'] = old['expanded_pool']
    manifest['combined_selection'] = selection
    manifest['combined_control'] = json.loads((cache/'control.json').read_text())
    manifest['strong_ranker_weight'] = weight
    manifest['reference_ranker'] = {k:old[k] for k in ['final_ranker','final_ranker_sha256']}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    directory = ROOT/'experiments/results/v8'
    directory.mkdir(parents=True,exist_ok=True)
    manifest['answer_file'] = (directory/'answer.csv').relative_to(ROOT).as_posix()
    vectors = np.load(ROOT/manifest['query_vectors'],allow_pickle=False)
    LEARNED_LOOKUP = dict(zip(sorted(set(queries.query_norm)),vectors))
    history = V5History(history_all)
    records = retrieve_expanded_features(queries,history,progress_every=500)
    features = [v5_features(q,r,history) for q,r in zip(queries.itertuples(index=False),records)]
    matrices = [np.column_stack([x,query_features(LEARNED_LOOKUP[text],ids,raw)])
        for text,(ids,raw),x in zip(queries.query_norm,records,features)]
    baseline = baseline_v5(records,features,final=True)
    winner = manifest['selection']['winner']
    model = joblib.load(ROOT/manifest['final_ranker'])
    scores = blend_scores(predict_scores(model,matrices,int(winner['trees'])),baseline,float(winner['weight']))
    if 0<weight<1:
        reference = joblib.load(ROOT/old['final_ranker'])
        reference_scores = blend_scores(predict_scores(reference,matrices,400),baseline,.75)
        scores = blend_scores(scores,reference_scores,weight)
    predictions = top50(records,scores)
    answer = pd.DataFrame({'query_id':queries.query_id.astype(str),'answer':[' '.join(ITEM_IDS[p]) for p in predictions]})
    assert len(answer)==2452 and answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    allowed=set(ITEM_IDS)
    for row in answer.itertuples(index=False):
        ids=row.answer.split(' ')
        assert len(row.query_id)==16 and len(ids)==len(set(ids))==50
        assert all(item in allowed and re.fullmatch('[0-9a-f]{16}',item) for item in ids)
    answer.to_csv(ROOT/manifest['answer_file'],index=False,encoding='utf-8',lineterminator='\n')
    manifest['answer_sha256']=sha256_file(ROOT/manifest['answer_file'])
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    notebook=json.loads((ROOT/'experiments/Avito_expanded_pool_v8_candidate.ipynb').read_text(encoding='utf-8'))
    title='поиск дообученной E5' if weight==0 else 'усиленный ранкер и поиск дообученной E5'
    notebook['cells'][0]['source']=[f'# Кандидат v8: {title}\n',
        '\nRun All создаёт experiments/results/v8/answer.csv.\n',
        'Описание экспериментов и ограничений — docs/EXPERIMENTS_V8.md. Метрика платформы пока неизвестна.\n']
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        value=''.join(cell['source'])
        if value.startswith('FROZEN_QUERY_MANIFEST = '):
            value='FROZEN_QUERY_MANIFEST = '+repr(manifest)+'\n'+value[value.index('\n')+1:]
        if value.startswith('def export_frozen_query_candidate():') and 0<weight<1:
            value=value.replace('    predictions=top50(records,score)', '''    reference_info=FROZEN_QUERY_MANIFEST['reference_ranker']
    reference_path=ROOT/reference_info['final_ranker']
    assert sha256_file(reference_path)==reference_info['final_ranker_sha256']
    reference_model=joblib.load(reference_path)
    matrices=[np.column_stack([x,y]) for x,y in zip(features,added)]
    reference_score=blend_scores(predict_scores(reference_model,matrices,400),original,.75)
    score=blend_scores(score,reference_score,FROZEN_QUERY_MANIFEST['strong_ranker_weight'])
    predictions=top50(records,score)''')
        ast.parse(value);cell['source']=value.splitlines(True)
        cell['outputs']=[];cell['execution_count']=None
    notebook['metadata']['frozen_query_selection']=manifest['selection']['winner']
    notebook['metadata']['v8_platform_recall50']=None
    explanation='''## Дополнительный канал кандидатогенерации v8

Дообученная E5 ищет по всем 189 212 объявлениям, а не только оценивает прежний
пул. Добавляем top-500 по cosine similarity и top-500 с географическим сигналом,
объединяем с прежними кандидатами без повторов и заново вычисляем признаки.
Параметры выбраны на development; разметка benchmark неизвестна.
В выбранном варианте вес нового ранкера равен нулю: прежний ранкер на этом
пуле показал лучший matched Recall@50. Ограничения — docs/EXPERIMENTS_V8.md.
'''
    if weight!=0:
        explanation=explanation[:explanation.index('В выбранном варианте')]+'Вес ранкеров выбран на development; см. docs/EXPERIMENTS_V8.md.\n'
    notebook['cells'].insert(-2,{'cell_type':'markdown','metadata':{},'source':explanation.splitlines(True),
        'id':hashlib.sha256(explanation.encode()).hexdigest()[:12]})
    (ROOT/'experiments/Avito_v8_candidate.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    assert sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH
    print('Selected v8 exported',manifest['answer_file'],manifest['answer_sha256'],flush=True)

if __name__=='__main__':export()
