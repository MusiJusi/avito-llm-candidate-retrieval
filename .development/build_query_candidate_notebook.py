"""Freeze full-train E5 inference in a self-contained Jupyter notebook."""
from pathlib import Path
import ast
import copy
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'artifacts/query-encoder-candidate/manifest.json').read_text())
document=json.loads((ROOT/'Avito_v0.5.ipynb').read_text(encoding='utf-8'))
cells=[]
for original in document['cells']:
    cell=copy.deepcopy(original)
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        if 'RUN_MODEL_SEARCH = ' in source and 'final_fit_and_export(' in source:
            source=source[:source.index('RUN_MODEL_SEARCH = ')]
        cell['source']=source.splitlines(True)
        cell['outputs']=[];cell['execution_count']=None
    cells.append(cell)
def add(kind,source):
    cell={'cell_type':kind,'metadata':{},'source':source.splitlines(True),
        'id':hashlib.sha256((kind+source).encode()).hexdigest()[:12]}
    if kind=='code':cell.update(outputs=[],execution_count=None)
    cells.append(cell)
cells[0]['source']=['# Отбор услуг: дообученный query-encoder E5\n',
    '\nRestart Kernel → Run All создаёт отдельный answer_query_encoder.csv.\n',
    'Веса query-encoder обучены на полном train; document-encoder и корпусные векторы E5 заморожены.\n',
    'Параметры выбраны на development; ограничения контроля сохранены в manifest.\n',
    'Основной answer.csv от v5 не меняется. Внешних inference API нет.\n',
    'Исследование и рецептура полного обучения — в Avito_neural_experiments.ipynb и .development.\n']
add('markdown','''## Зафиксированный нейросетевой кандидат

Документы кодируются исходной multilingual-e5-small, запросы — отдельно
дообученной E5. Их векторы сравниваются внутри полного пула v5. Нейросетевой
score дополняет географию и остальные признаки. OOF-рецептура и аудит исключения
собственных меток сохранены вместе с исходниками; обычное воспроизведение
использует поставляемые финальные веса и детерминированные query-векторы.
''')
add('code','FROZEN_QUERY_MANIFEST = '+repr(manifest)+'''
V5_CACHE = CACHE
V5_FP = fingerprint
V5_MANIFEST = json.loads((V5_CACHE/'manifest.json').read_text(encoding='utf-8'))
V5_WINNER = V5_MANIFEST['selection']['winner']
assert FROZEN_QUERY_MANIFEST['input_sha256'] == input_hashes
QE_CONFIG = {'max_query_length':64}
encoder_path = ROOT/FROZEN_QUERY_MANIFEST['encoder']
assert sha256_file(encoder_path/'model.safetensors') == FROZEN_QUERY_MANIFEST['encoder_sha256']
''')
for filename,names in [('microcat_v6.py',{'baseline_v5'}),
    ('query_encoder_pilot.py',{'encode_learned'}),('query_encoder_rank.py',{'query_features'})]:
    source=(ROOT/'.development'/filename).read_text(encoding='utf-8')
    tree=ast.parse(source)
    add('code','\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)+'\n')
add('code','''def export_frozen_query_candidate():
    vector_path=ROOT/FROZEN_QUERY_MANIFEST['query_vectors']
    if USE_CACHE and vector_path.exists():
        assert sha256_file(vector_path)==FROZEN_QUERY_MANIFEST['query_vectors_sha256']
        vectors=np.load(vector_path,allow_pickle=False)
    else:
        model=AutoModel.from_pretrained(encoder_path,local_files_only=True,attn_implementation='eager').to(DEVICE)
        tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
        vectors=encode_learned(model,tokenizer,sorted(set(queries.query_norm)))
        del model,tokenizer
        gc.collect()
        if DEVICE=='cuda':torch.cuda.empty_cache()
    assert vectors.shape==(queries.query_norm.nunique(),384) and np.isfinite(vectors).all()
    lookup={text:i for i,text in enumerate(sorted(set(queries.query_norm)))}
    pool_path=V5_CACHE/f'benchmark_{V5_FP}.joblib'
    if USE_CACHE and pool_path.exists():
        records,features=joblib.load(pool_path)
    else:
        history=V5History(history_all)
        records=retrieve_semantic_features(queries,history,progress_every=500)
        features=[v5_features(query,record,history) for query,record in zip(queries.itertuples(index=False),records)]
        del history
    original=baseline_v5(records,features,final=True)
    added=[query_features(vectors[lookup[text]],ids,base) for text,(ids,base) in zip(queries.query_norm,records)]
    winner=FROZEN_QUERY_MANIFEST['selection']['winner']
    if winner['variant']=='direct_query_blend':
        score=blend_scores([x[:,1] for x in added],original,float(winner['weight']))
    else:
        path=ROOT/FROZEN_QUERY_MANIFEST['final_ranker']
        assert sha256_file(path)==FROZEN_QUERY_MANIFEST['final_ranker_sha256']
        model=joblib.load(path)
        assert model.n_features_in_==51
        score=blend_scores(predict_scores(model,[np.column_stack([x,y]) for x,y in zip(features,added)],int(winner['trees'])),original,float(winner['weight']))
    predictions=top50(records,score)
    answer=pd.DataFrame({'query_id':queries.query_id.astype(str),'answer':[' '.join(ITEM_IDS[prediction]) for prediction in predictions]})
    assert list(answer.columns)==['query_id','answer'] and len(answer)==len(queries)==2452
    assert answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    allowed=set(ITEM_IDS)
    for row in answer.itertuples(index=False):
        assert len(row.query_id)==16
        ids=row.answer.split(' ')
        assert len(ids)==50 and len(set(ids))==50
        assert all(re.fullmatch('[0-9a-f]{16}',item) and item in allowed for item in ids)
    output=ROOT/FROZEN_QUERY_MANIFEST['answer_file']
    answer.to_csv(output,index=False,encoding='utf-8',lineterminator='\\n')
    assert sha256_file(output)==FROZEN_QUERY_MANIFEST['answer_sha256']
    print('Validated neural candidate:',output,'SHA256:',sha256_file(output))
    return answer

query_answer=export_frozen_query_candidate()
query_answer.head()
''')
document['cells']=cells
document['metadata']['frozen_query_selection']=manifest['selection']['winner']
for cell in cells:
    if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
(ROOT/'Avito_query_encoder_candidate.ipynb').write_text(json.dumps(document,ensure_ascii=False,indent=1),encoding='utf-8')
print('Created self-contained learned-query candidate notebook.')
