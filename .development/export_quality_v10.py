"""Refit the development winner and export one reproducible v10 candidate."""
from pathlib import Path
import ast
import gc
import json
import subprocess
import sys

EXPORT_V10=Path(__file__).resolve()
source=EXPORT_V10.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(EXPORT_V10)
from activate_warm_history import activate
activate(globals())


def prepare_final_field_model(fit_selected=True):
    path=ROOT/'artifacts/field-ranker-v10/selection.json'
    if not path.exists():return {}
    selection=json.loads(path.read_text())
    if selection['winner']['weight']==0:return {}
    source=ROOT/'.development/field_ranker_v10.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),globals())
    globals()['FIELD_RANK_DRIVER']=source
    globals()['FIELD_RANK_CACHE']=ROOT/'artifacts/field-ranker-v10'
    globals()['FIELD_RANK_FP']=selection['fingerprint']
    globals()['FIELD_RANK_NAMES']=selection['feature_names'][-8:]
    globals()['FIELD_DOCUMENTS']=prepare_fields()
    model_path=None
    if fit_selected:
        data=field_training('final');fit_field(data,'final')
        del data;gc.collect()
        model_path=FIELD_RANK_CACHE/f'final_ranker_{FIELD_RANK_FP}.joblib'
    texts=sorted(set(queries.query_norm))
    teacher_path=V10_CACHE/f'benchmark_field_teachers_{V10_FP}.joblib'
    save_cache({'texts':texts,'vectors':np.stack([
        semantic_index.queries[semantic_index.query_to_row[t]] for t in texts])},teacher_path)
    return dict(field_selection=selection,field_ranker=model_path.relative_to(ROOT).as_posix() if model_path else None,
        field_ranker_sha256=sha256_file(model_path) if model_path else None,field_weight=selection['winner']['weight'],
        field_teacher_vectors=teacher_path.relative_to(ROOT).as_posix(),field_teacher_vectors_sha256=sha256_file(teacher_path),
        field_documents=json.loads((ROOT/'artifacts/semantic-fields-v10/vectors.json').read_text()))


def prepare_final_bge_model():
    path=ROOT/'artifacts/bge-m3-v10/selection.json'
    if not path.exists():return {}
    selection=json.loads(path.read_text())
    if selection['winner']['kind']=='baseline':return {}
    source=ROOT/'.development/bge_ranker_v10.py'
    definitions=[n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),globals())
    globals()['BGE_DRIVER']=source;globals()['BGE_CACHE']=ROOT/'artifacts/bge-m3-v10'
    globals()['BGE_FP']=selection['fingerprint']
    globals()['BGE_NAMES']=['bge_query_cosine','bge_query_delta_e5','bge_query_filter_cosine','bge_query_filter_delta_e5']
    install_field_functions()
    globals()['BGE_DOCUMENTS'],globals()['BGE_QUERIES']=prepare_bge()
    ranker_path=None
    if selection['winner']['kind']=='ranker':
        data=bge_training('final');fit_bge(data,'final');del data;gc.collect()
        ranker_path=BGE_CACHE/f'final_ranker_{BGE_FP}.joblib'
    texts=sorted(set(queries.query_norm)|{query_filter_text(q,f)
        for q,f in queries[['query_norm','search_infm_params_text']].itertuples(index=False,name=None)})
    vectors_path=BGE_CACHE/f'benchmark_vectors_{BGE_FP}.joblib'
    save_cache({'texts':texts,'vectors':np.stack([BGE_QUERIES[t] for t in texts])},vectors_path)
    vectors_manifest=json.loads((BGE_CACHE/'vectors.json').read_text())
    return {'bge_selection':selection,'bge_kind':selection['winner']['kind'],
        'bge_weight':selection['winner']['weight'],'bge_documents':vectors_manifest['documents'],
        'bge_trees':selection['trees'],
        'bge_queries':vectors_path.relative_to(ROOT).as_posix(),'bge_queries_sha256':sha256_file(vectors_path),
        'bge_ranker':ranker_path.relative_to(ROOT).as_posix() if ranker_path else None,
        'bge_ranker_sha256':sha256_file(ranker_path) if ranker_path else None,
        'bge_checkpoint':vectors_manifest['model'],'bge_temperature':SEMANTIC_CONFIG['temperature']}


def export_quality():
    global RESOURCES,CONTEXT_VECTORS,LEARNED_LOOKUP,GEO_EVIDENCE,QUALITY_BANK
    with (V10_CACHE/'warm_joint.log').open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',str(ROOT/'.development/warm_joint_v10.py')],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    # The selected history auxiliary has a documented control veto. Apply it
    # before judging the cross-encoder against the final upstream ensemble.
    subprocess.run([sys.executable,'-u',str(ROOT/'.development/apply_warm_control_veto.py')],
        cwd=ROOT,check=True)
    ce_cache=ROOT/'artifacts/cross-finetune-v10'
    ce_pilot=ce_cache/'selection.json'
    if ce_pilot.exists() and json.loads(ce_pilot.read_text())['winner']['weight']>0:
        # Compare the trained pilot with the now-fixed upstream ensemble first.
        # Run before fitting/materializing final pools, keeping laptop RAM bounded.
        with (V10_CACHE/'cross_joint.log').open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',str(ROOT/'.development/cross_joint_v10.py')],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    with (V10_CACHE/'error_analysis.log').open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',str(ROOT/'.development/analyze_quality_errors_v10.py')],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    selection=json.loads((V10_CACHE/'selection.json').read_text())
    assert selection['fingerprint']==V10_FP
    winner=selection['winner']
    assert winner['matched_recall50']>=.9540428139856304
    install_retriever();RESOURCES=prepare_resources()
    CONTEXT_VECTORS,contextual_source=contextual_vectors()
    name=winner['variant']
    model_path=None
    field_path=ROOT/'artifacts/field-ranker-v10/selection.json'
    field_choice=json.loads(field_path.read_text()) if field_path.exists() else None
    need_field=bool(field_choice and field_choice['winner']['weight']>0)
    bge_path=ROOT/'artifacts/bge-m3-v10/selection.json'
    bge_choice=json.loads(bge_path.read_text()) if bge_path.exists() else None
    need_bge_rank=bool(bge_choice and bge_choice['winner']['kind']=='ranker')
    bge_replaces=bool(need_bge_rank and bge_choice['winner']['weight']==1.)
    primary_replaced=bge_replaces or bool(need_field and field_choice['winner']['weight']==1.)
    if name in MODEL_RECIPES or need_field or need_bge_rank:
        data=prepare_quality_training('final')
        if name in MODEL_RECIPES and not primary_replaced:
            fit_quality(data,'final',[name])
            model_path=V10_CACHE/f'final_{name}_{V10_FP}.joblib'
        del data;gc.collect()
    # Fit before materializing the full benchmark pools to keep peak RAM bounded.
    field_manifest=prepare_final_field_model(fit_selected=not bge_replaces) if need_field else {}
    bge_manifest=prepare_final_bge_model()
    aux_selection=json.loads((V10_CACHE/'warm_aux_selection.json').read_text())
    aux_manifest={}
    if aux_selection['winner']['weight']>0:
        aux_name=aux_selection['winner']['variant']
        fit_quality(None,'final',[aux_name])
        aux_path=V10_CACHE/f'final_{aux_name}_{V10_FP}.joblib'
        aux_manifest={'warm_aux_selection':aux_selection,
            'warm_aux_ranker':aux_path.relative_to(ROOT).as_posix(),'warm_aux_ranker_sha256':sha256_file(aux_path),
            'warm_aux_weight':aux_selection['winner']['weight'],'warm_aux_trees':MODEL_RECIPES[aux_name]['trees']}
    original=json.loads((V9_CACHE/'manifest.json').read_text())
    bank=joblib.load(ROOT/original['history_bank']);GEO_EVIDENCE=bank['geo']
    vectors=np.load(ROOT/original['query_vectors'],allow_pickle=False)
    LEARNED_LOOKUP=dict(zip(sorted(set(queries.query_norm)),vectors))
    bundle=joblib.load(ROOT/original['query_evidence'])
    lookup={text:i for i,text in enumerate(bundle['texts'])}
    evidence_rows=tuple(array[[lookup[t] for t in queries.query_norm]] for array in bundle['results'])
    history=V5History(history_all)
    records=retrieve_context_features(queries,history,progress_every=500)
    features=[v5_features(q,r,history) for q,r in zip(queries.itertuples(index=False),records)]
    all_features=matrices(queries,records,features,evidence_rows)
    baseline=baseline_v5(records,features,final=True)
    reference=joblib.load(ROOT/original['final_ranker'])
    scores=blend_scores(predict_scores(reference,[x[:,:51] for x in all_features],400),baseline,.75)
    old_model=joblib.load(ROOT/original['context_ranker'])
    scores=blend_scores(predict_scores(old_model,all_features,400),scores,.25)
    quality_bank=(QualityEvidenceWithTrace if name.startswith('warm_') or aux_manifest else QualityEvidence)(history_all)
    quality_bank_path=V10_CACHE/f'final_history_bank_{V10_FP}.joblib'
    save_cache(quality_bank,quality_bank_path)
    texts=sorted(set(query_filter_text(q,f) for q,f in queries[['query_norm','search_infm_params_text']].itertuples(index=False,name=None)))
    query_vectors_path=V10_CACHE/f'benchmark_contextual_vectors_{V10_FP}.joblib'
    save_cache({'texts':texts,'vectors':np.stack([CONTEXT_VECTORS[text] for text in texts]),
        'producer':'original frozen E5 with query and filters','source':contextual_source.relative_to(ROOT).as_posix()},query_vectors_path)
    manifest={'fingerprint':V10_FP,'selection':selection,'control':json.loads((V10_CACHE/'control.json').read_text()),
        'category_boost':selection['category_boost'],'weight':winner['weight'],
        'columns':MODEL_RECIPES[name]['columns'] if model_path else None,
        'trees':MODEL_RECIPES[name]['trees'] if model_path else None,
        'history_bank':quality_bank_path.relative_to(ROOT).as_posix(),
        'history_bank_sha256':sha256_file(quality_bank_path),
        'contextual_vectors':query_vectors_path.relative_to(ROOT).as_posix(),
        'contextual_vectors_sha256':sha256_file(query_vectors_path),
        'ranker':model_path.relative_to(ROOT).as_posix() if model_path else None,
        'ranker_sha256':sha256_file(model_path) if model_path else None,
        'module_sha256':{f'.development/{filename}':sha256_file(ROOT/'.development'/filename)
            for filename in ['quality_signals_v10.py','quality_model_v10.py','quality_trace_v10.py']},
        'feature_names':NEW_FEATURE_NAMES+(TRACE_NAMES if name.startswith('warm_') else []),
        'primary_feature_width':86 if name.startswith('warm_') else 82,
        'unused_primary_ranker_omitted':primary_replaced,
        'baseline_version':'v9 platform 0.900753'}
    manifest.update(field_manifest)
    manifest.update(bge_manifest)
    manifest.update(aux_manifest)
    joint_path=ce_cache/'joint_selection.json'
    if joint_path.exists():
        joint=json.loads(joint_path.read_text())
        if joint['winner']['weight']>0:
            model_directory=ROOT/joint['model']
            tokenizer_directory=ROOT/'models/mmarco-miniLM-cross-encoder'
            files=[p for p in model_directory.iterdir() if p.suffix in {'.json','.safetensors'}]
            files.extend(p for p in tokenizer_directory.iterdir() if p.name!='model.safetensors' and p.suffix in {'.json','.model','.txt'})
            manifest.update(cross_selection=joint,cross_weight=joint['winner']['weight'],
                cross_model=model_directory.relative_to(ROOT).as_posix(),
                cross_weights=(model_directory/'model.safetensors').relative_to(ROOT).as_posix(),
                cross_weights_sha256=sha256_file(model_directory/'model.safetensors'),
                cross_tokenizer=tokenizer_directory.relative_to(ROOT).as_posix(),
                cross_files_sha256={p.relative_to(ROOT).as_posix():sha256_file(p) for p in files},
                cross_inference='Local CUDA BF16 eager attention, CPU float32 fallback; texts evaluated by the actual model')
            manifest['module_sha256']['.development/quality_cross_model_v10.py']=sha256_file(ROOT/'.development/quality_cross_model_v10.py')
    from quality_model_v10 import adjust_quality_scores
    ce_reference=scores
    scores=adjust_quality_scores(ROOT,manifest,queries,items,records,all_features,scores,
        semantic_index.items,sha256_file,blend_scores,predict_scores)
    from quality_cross_model_v10 import adjust_cross_scores
    scores=adjust_cross_scores(ROOT,manifest,queries,items,records,scores,ce_reference,
        sha256_file,blend_scores,stable_topk)
    answer=pd.DataFrame({'query_id':queries.query_id.astype(str),'answer':[' '.join(ITEM_IDS[p]) for p in top50(records,scores)]})
    directory=ROOT/'experiments/results/v10';directory.mkdir(parents=True,exist_ok=True)
    output=directory/'answer.csv';answer.to_csv(output,index=False,encoding='utf-8',lineterminator='\n')
    assert len(answer)==2452 and answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    for row in answer.itertuples(index=False):
        ids=row.answer.split(' ')
        assert len(row.query_id)==16 and len(ids)==len(set(ids))==50
        assert all(v in ITEM_TO_ROW and re.fullmatch('[0-9a-f]{16}',v) for v in ids)
    manifest.update(answer_file=output.relative_to(ROOT).as_posix(),answer_sha256=sha256_file(output))
    (V10_CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    notebook=json.loads((ROOT/'experiments/Avito_v9_candidate.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]['source']=['# Avito v10: расширенное обучение и контекст запроса\n',
        '\nRun All воспроизводит experiments/results/v10/answer.csv.\n',
        'Версия v9 (0.900753 на платформе) сохранена отдельно. Измерения: docs/EXPERIMENTS_V10.md.\n']
    notebook['cells'].insert(len(notebook['cells'])-1,{'cell_type':'code','metadata':{},'execution_count':None,
        'outputs':[],'id':'quality-v10-manifest','source':('QUALITY_V10_MANIFEST = '+repr(manifest)+'\n').splitlines(True)})
    notebook['cells'].insert(len(notebook['cells'])-2,{'cell_type':'markdown','metadata':{},'id':'quality-v10-description',
        'source':['## Дополнительный отбор v10\n','\nКатегория используется мягко; 0 не означает ограничение. ',
            'Новый ранкер учитывает E5-вектор текста запроса с фильтрами, историю объявления ',
            'и географические переходы с учётом вида услуги. Обучение и контроль разделены; ',
            'ранее просмотренная валидация имеет ограничения, описанные в отчёте.\n']})
    frozen=dict(original,answer_file=manifest['answer_file'],answer_sha256=manifest['answer_sha256'])
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        text=''.join(cell['source'])
        if text.startswith('FROZEN_QUERY_MANIFEST = '):
            text='FROZEN_QUERY_MANIFEST = '+repr(frozen)+'\n'+text[text.index('\n')+1:]
        if text.startswith('def export_frozen_query_candidate():'):
            addition='''    ce_reference=score
    from quality_model_v10 import adjust_quality_scores
    score=adjust_quality_scores(ROOT,QUALITY_V10_MANIFEST,queries,items,records,context_matrices,score,
        semantic_index.items,sha256_file,blend_scores,predict_scores)
    from quality_cross_model_v10 import adjust_cross_scores
    score=adjust_cross_scores(ROOT,QUALITY_V10_MANIFEST,queries,items,records,score,ce_reference,
        sha256_file,blend_scores,stable_topk)
    predictions=top50(records,score)'''
            assert text.count('    predictions=top50(records,score)')==1
            text=text.replace('    predictions=top50(records,score)',addition)
        ast.parse(text);cell['source']=text.splitlines(True);cell['outputs']=[];cell['execution_count']=None
    notebook['metadata']['v10_platform_recall50']=None
    (ROOT/'experiments/Avito_v10_candidate.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    assert sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH
    assert sha256_file(ROOT/original['answer_file'])==original['answer_sha256']
    print('Quality v10 exported',manifest['answer_file'],manifest['answer_sha256'],flush=True)


if __name__=='__main__':export_quality()
