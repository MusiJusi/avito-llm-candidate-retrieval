"""Refit a development-selected context model and freeze offline inference."""
from pathlib import Path
import ast
import gc
import json
import hashlib

EXPORT_CONTEXT_DRIVER=Path(__file__).resolve()
source=EXPORT_CONTEXT_DRIVER.with_name('context_rank_v9.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(EXPORT_CONTEXT_DRIVER)

def export():
    global RESOURCES,LEARNED_LOOKUP,GEO_EVIDENCE
    selection=json.loads((CONTEXT_CACHE/'selection.json').read_text())
    winner=selection['winner'];variant=winner['variant']
    if variant=='confirmed_v8_pool':
        print('No development improvement: retain the already reproduced v8 answer.',flush=True)
        return
    assert winner['matched_recall50']>.9528152912874686
    retriever_source=install_retriever()
    RESOURCES=prepare_resources()
    if variant in FEATURE_SETS:
        data=prepare_training('final')
        model=fit_models(data,'final',[variant])[variant]
        del data;gc.collect()
        ranker_path=CONTEXT_CACHE/f'final_{variant}_{CONTEXT_FP}.joblib'
    else:ranker_path=None
    original=json.loads((ROOT/'artifacts/combined-pool-v8/manifest.json').read_text())
    vectors=np.load(ROOT/original['query_vectors'],allow_pickle=False)
    LEARNED_LOOKUP=dict(zip(sorted(set(queries.query_norm)),vectors))
    coordinates,document_ids,documents,teacher_lookup=RESOURCES
    GEO_EVIDENCE=GeographicEvidence(history_all,coordinates,{k:v for k,v in centers.iterrows()})
    semantic_bank=SemanticEvidence(history_all,teacher_lookup,document_ids,documents,MICROS)
    bank_path=CONTEXT_CACHE/f'history_bank_{CONTEXT_FP}.joblib'
    save_cache({'geo':GEO_EVIDENCE,'semantic':semantic_bank},bank_path)
    texts=sorted(set(queries.query_norm))
    results=semantic_bank.query_evidence(np.stack([teacher_lookup[text] for text in texts]),CONTEXT_CONFIG['neighbors'])
    signal_path=CONTEXT_CACHE/f'query_evidence_{CONTEXT_FP}.joblib'
    save_cache({'texts':texts,'results':results},signal_path)
    lookup={text:i for i,text in enumerate(texts)}
    ordered=tuple(array[[lookup[text] for text in queries.query_norm]] for array in results)
    history=V5History(history_all)
    records=retrieve_context_features(queries,history,progress_every=500)
    features=[v5_features(q,r,history) for q,r in zip(queries.itertuples(index=False),records)]
    all_features=matrices(queries,records,features,ordered)
    baseline=baseline_v5(records,features,final=True)
    reference=joblib.load(ROOT/original['final_ranker'])
    scores=blend_scores(predict_scores(reference,[x[:,:51] for x in all_features],400),baseline,.75)
    if ranker_path is not None:
        selected=[np.ascontiguousarray(x[:,FEATURE_SETS[variant]]) for x in all_features]
        scores=blend_scores(predict_scores(model,selected,400),scores,float(winner['weight']))
    answer=pd.DataFrame({'query_id':queries.query_id.astype(str),
        'answer':[' '.join(ITEM_IDS[p]) for p in top50(records,scores)]})
    directory=ROOT/'experiments/results/v9';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'answer.csv';answer.to_csv(path,index=False,encoding='utf-8',lineterminator='\n')
    assert len(answer)==2452 and answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    allowed=set(ITEM_IDS)
    for row in answer.itertuples(index=False):
        ids=row.answer.split(' ')
        assert len(row.query_id)==16 and len(ids)==len(set(ids))==50
        assert all(item in allowed and re.fullmatch('[0-9a-f]{16}',item) for item in ids)
    manifest=dict(original)
    manifest.update(fingerprint=CONTEXT_FP,answer_file=path.relative_to(ROOT).as_posix(),
        answer_sha256=sha256_file(path),context_selection=selection,
        context_control=json.loads((CONTEXT_CACHE/'control.json').read_text()),
        context_ranker=ranker_path.relative_to(ROOT).as_posix() if ranker_path else None,
        context_ranker_sha256=sha256_file(ranker_path) if ranker_path else None,
        context_columns=FEATURE_SETS.get(variant),context_feature_names=FEATURE_NAMES,
        history_bank=bank_path.relative_to(ROOT).as_posix(),history_bank_sha256=sha256_file(bank_path),
        query_evidence=signal_path.relative_to(ROOT).as_posix(),query_evidence_sha256=sha256_file(signal_path),
        context_module_sha256=sha256_file(ROOT/'.development/context_signals_v9.py'))
    (CONTEXT_CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    notebook=json.loads((ROOT/'experiments/Avito_v8_candidate.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]['source']=['# Кандидат v9: география и семантическая история\n',
        '\nRun All создаёт experiments/results/v9/answer.csv.\n',
        'Измерения, обучение и ограничения: docs/EXPERIMENTS_V9.md. Метрика платформы пока неизвестна.\n']
    descriptions={
        1:'''## Базовые компоненты и воспроизведение

Notebook содержит определения наших лексических, семантических и исторических
компонентов. Обучение по умолчанию выключено. Последняя ячейка использует
зафиксированные веса и создаёт CSV, указанный в манифесте.
Исходные Parquet, модели и результат проверяются по SHA-256.
`search_category` участвует в определении контекста для изоляции истории,
но не передаётся scoring. Рецептуры обучения находятся в .development и
experiments/Avito_neural_experiments.ipynb.
''',
        22:'''## Базовые исторические сигналы

Воспроизводятся сохранённые компоненты v5–v7. У старых вспомогательных priors v4
есть известное ограничение независимости; оно отражено в отчёте. Новые признаки
v9 обучены на истории с исключением собственных OOF-контекстов и проверкой
совпадения истории с историей соответствующего query-encoder.
Финальные исторические банки строятся только по предоставленному train.
''',
        32:'''## Контроль и ограничения оценки

Development и контроль из 600 запросов ранее просматривались. Они не являются
новым независимым тестом. Конфигурация выбирается по development; её значение
на контроле сообщается отдельно. Оценка на платформе для v9 пока неизвестна.
Измерения и конкретные ошибки приведены в docs/EXPERIMENTS_V9.md.
'''}
    for index,text in descriptions.items():
        assert notebook['cells'][index]['cell_type']=='markdown'
        notebook['cells'][index]['source']=text.splitlines(True)
    description=(f'## Дополнительный отбор v9\n\n'
        f'Выбрана конфигурация `{variant}` с весом {winner["weight"]}. '
        'Новые сигналы используют историческую географию и семантические соседства. '
        'Собственные контрольные метки в эти признаки не попадают. '
        'Отбор дополняет базовую комбинацию E5 и LambdaRank через RRF. '
        'Банки истории и сохранённые результаты модельного расчёта для текстов '
        'запросов не содержат тестовой разметки.\n')
    notebook['cells'].insert(len(notebook['cells'])-1,{'cell_type':'markdown',
        'metadata':{},'source':description.splitlines(True),'id':'context-v9-inference'})
    # Replace the existing extra retrieval definition with our extended version.
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        value=''.join(cell['source'])
        if value.startswith('FROZEN_QUERY_MANIFEST = '):
            value='FROZEN_QUERY_MANIFEST = '+repr(manifest)+'\n'+value[value.index('\n')+1:]
        if 'def retrieve_expanded_features(' in value:
            value='LEARNED_LOOKUP = {}\nGEO_EVIDENCE = None\n\n'+retriever_source+'\n'
        if value.startswith('def export_frozen_query_candidate():'):
            value=value.replace('global LEARNED_LOOKUP','global LEARNED_LOOKUP, GEO_EVIDENCE')
            marker="    pool_path=V5_CACHE/f'expanded_benchmark_{FROZEN_QUERY_MANIFEST[\"fingerprint\"]}.joblib'"
            initialization='''    # The frozen bank contains training aggregates, not benchmark answers.
    import sys
    sys.path.insert(0,str(ROOT/'.development'))
    from context_signals_v9 import SemanticEvidence
    assert sha256_file(ROOT/'.development/context_signals_v9.py')==FROZEN_QUERY_MANIFEST['context_module_sha256']
    bank_path=ROOT/FROZEN_QUERY_MANIFEST['history_bank']
    assert sha256_file(bank_path)==FROZEN_QUERY_MANIFEST['history_bank_sha256']
    bank=joblib.load(bank_path);GEO_EVIDENCE=bank['geo']
    evidence_path=ROOT/FROZEN_QUERY_MANIFEST['query_evidence']
    assert sha256_file(evidence_path)==FROZEN_QUERY_MANIFEST['query_evidence_sha256']
    evidence=joblib.load(evidence_path)
    neighbor_lookup={text:i for i,text in enumerate(evidence['texts'])}
    neighbor_results=tuple(array[[neighbor_lookup[text] for text in queries.query_norm]] for array in evidence['results'])
'''
            assert marker in value
            value=value.replace(marker,initialization+marker)
            value=value.replace('retrieve_expanded_features(queries,history,progress_every=500)',
                'retrieve_context_features(queries,history,progress_every=500)')
            extension='''    context_matrices=[]
    for row,(query,(ids,raw),base,learned) in enumerate(zip(queries.itertuples(index=False),records,features,added)):
        geography=GEO_EVIDENCE.features(query.search_location_id,items.item_latitude.to_numpy()[ids],items.item_longitude.to_numpy()[ids])
        neighborhood=SemanticEvidence.features(neighbor_results[0][row],neighbor_results[1][row],neighbor_results[2][row],
            semantic_index.items[ids],ITEM_MICRO_COLS[ids],rankdata)
        context_matrices.append(np.column_stack([base,learned,geography,neighborhood]).astype(np.float32))
    context_path=FROZEN_QUERY_MANIFEST['context_ranker']
    if context_path:
        assert sha256_file(ROOT/context_path)==FROZEN_QUERY_MANIFEST['context_ranker_sha256']
        context_model=joblib.load(ROOT/context_path)
        columns=FROZEN_QUERY_MANIFEST['context_columns']
        new_scores=predict_scores(context_model,[np.ascontiguousarray(x[:,columns]) for x in context_matrices],400)
        score=blend_scores(new_scores,score,FROZEN_QUERY_MANIFEST['context_selection']['winner']['weight'])
    predictions=top50(records,score)'''
            value=value.replace('    predictions=top50(records,score)',extension)
        ast.parse(value);cell['source']=value.splitlines(True);cell['outputs']=[];cell['execution_count']=None
    notebook['metadata']['v9_platform_recall50']=None
    (ROOT/'experiments/Avito_v9_candidate.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    assert sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH
    print('Context candidate exported',manifest['answer_file'],manifest['answer_sha256'],flush=True)

if __name__=='__main__':export()
