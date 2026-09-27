"""Refit the selected E5 candidate using full train and export a separate CSV.

The original document encoder and v5 model stay frozen. Full-train query vectors
are supplied for offline inference, alongside the learned query-encoder weights.
Only if development chose the learned ranker do we repeat its six OOF models for
final ranker training; the direct-score candidate does not require this cost.
"""
from pathlib import Path
import ast
import hashlib
import json
import gc

EXPORT_DRIVER=Path(__file__).resolve()
driver=EXPORT_DRIVER.with_name('query_encoder_rank.py')
tree=ast.parse(driver.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(driver)
exec(compile(tree,str(driver),'exec'),globals())
__file__=str(EXPORT_DRIVER)
EXPORT_CACHE=ROOT/'artifacts/query-encoder-candidate'
EXPORT_CACHE.mkdir(exist_ok=True)
EXPORT_FP=hashlib.sha256((QR_FP+hashlib.sha256(EXPORT_DRIVER.read_bytes()).hexdigest()).encode()).hexdigest()[:16]


def full_document_vectors():
    ids=sorted(history_all.item_id.unique())
    path=EXPORT_CACHE/f'documents_{EXPORT_FP}.npy'
    if USE_CACHE and path.exists():return ids,np.load(path,allow_pickle=False)
    # A report-path repair changes storage code, not the frozen document vectors.
    # Reuse only an alias whose complete-data provenance and content hash match.
    alias=EXPORT_CACHE/f'document_vectors_{QE_FP}.npy'
    report=EXPORT_CACHE/'training_report.json'
    if USE_CACHE and alias.exists() and report.exists():
        declared=json.loads(report.read_text(encoding='utf-8'))
        if declared.get('input_sha256')==input_hashes and declared.get('full_train_refit'):
            assert sha256_file(alias)==declared['document_vectors_sha256']
            vectors=np.load(alias,allow_pickle=False)
            assert vectors.shape==(len(ids),384)
            os.link(alias,path)
            return ids,vectors
    old_ids,old_vectors=training_documents()
    old_lookup={item:i for i,item in enumerate(old_ids)}
    vectors=np.empty((len(ids),384),np.float32)
    missing=[];positions=[]
    for row,item in enumerate(ids):
        if item in old_lookup:vectors[row]=old_vectors[old_lookup[item]]
        elif item in ITEM_TO_ROW:vectors[row]=semantic_index.items[ITEM_TO_ROW[item]]
        else:missing.append(item);positions.append(row)
    if missing:
        metadata=pd.read_parquet(ROOT/'train.parquet',columns=['item_id','item_title_raw','item_description_raw','item_infm_params_text'])
        metadata=metadata.drop_duplicates('item_id').set_index('item_id')
        documents=metadata.loc[missing].fillna('')
        texts=(documents.item_title_raw+'. '+documents.item_infm_params_text.str.slice(0,SEMANTIC_CONFIG['params_chars'])+
            '. '+documents.item_description_raw.str.slice(0,SEMANTIC_CONFIG['description_chars'])).tolist()
        vectors[positions]=semantic_index.encode(texts,'passage: ')
        semantic_index.encoder=semantic_index.tokenizer=None
        del metadata,documents,texts
        gc.collect()
        if DEVICE=='cuda':torch.cuda.empty_cache()
    save_array(vectors,path)
    return ids,vectors


def fit_final_encoder(ids,documents):
    # Same optimizer, seed, teacher and loss as the selected development pilot.
    # Its helper saves epochs 1 and 3; history is replaced only for this full refit.
    global training_history,QE_CACHE,all_evaluation,development
    old_history,old_cache,old_all,old_dev=training_history,QE_CACHE,all_evaluation,development
    original_function=globals()['training_documents']
    try:
        training_history=history_all
        QE_CACHE=EXPORT_CACHE
        all_evaluation=development.iloc[:0]
        development=queries
        globals()['training_documents']=lambda:(ids,documents)
        # The pilot report hashes this alias after training, even when vectors
        # came from the separately fingerprinted full-train cache above.
        alias=EXPORT_CACHE/f'document_vectors_{QE_FP}.npy'
        if not alias.exists():os.link(EXPORT_CACHE/f'documents_{EXPORT_FP}.npy',alias)
        train_query_encoder()
    finally:
        training_history,QE_CACHE,all_evaluation,development=old_history,old_cache,old_all,old_dev
        globals()['training_documents']=original_function
    model_path=EXPORT_CACHE/f'epoch_3_{QE_FP}'
    vector_path=EXPORT_CACHE/f'queries_epoch_3_{QE_FP}.npy'
    assert model_path.exists() and vector_path.exists()
    # The pilot's development field is not an evaluation split in full refit:
    # it was temporarily used only to request benchmark query-vector inference.
    report_path=EXPORT_CACHE/'training_report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'))
        report.update(full_train_refit=True,training_completed=True,input_sha256=input_hashes,
            training_pairs_available=len(history_all),training_query_texts=history_all.query_norm.nunique(),
            sampled_pairs=int(history_all.query_norm.nunique()*QE_CONFIG['epochs']),
            development_text_overlap=None,benchmark_query_text_seen_fraction=float(queries.query_norm.isin(history_all.query_norm).mean()))
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return model_path,vector_path


def fit_final_query_ranker(ids,documents,trees):
    global training,training_history,QR_CACHE,QR_FP
    old_frame,old_history,old_cache,old_fp=training,training_history,QR_CACHE,QR_FP
    original_function=globals()['training_documents']
    try:
        training=history_all[history_all.context_key.isin(gold)].drop_duplicates('context_key')
        training=training[[*QUERY_COLS,'query_norm','context_key']].sort_values('context_key').reset_index(drop=True)
        training_history=history_all
        QR_CACHE=EXPORT_CACHE
        QR_FP=EXPORT_FP
        globals()['training_documents']=lambda:(ids,documents)
        # Change cache stage only; preserve the exact sampling and fold exclusions.
        class FinalStage(ast.NodeTransformer):
            def visit_Constant(self,node):
                if isinstance(node.value,str):node.value=node.value.replace('evaluation_','final_')
                return node
        rank_source=EXPORT_DRIVER.with_name('query_encoder_rank.py')
        source=ast.parse(rank_source.read_text(encoding='utf-8'))
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_query_oof')
        function=FinalStage().visit(function)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(rank_source),'exec'),globals())
        data,extra=prepare_query_oof()
        path=EXPORT_CACHE/f'final_ranker_{EXPORT_FP}.joblib'
        if USE_CACHE and path.exists():return joblib.load(path),path
        x=np.column_stack([data['X'],extra])
        model=lgb.LGBMRanker(n_estimators=trees,num_leaves=31,learning_rate=.05,max_bin=127,
            min_child_samples=50,reg_lambda=10,random_state=SEED,n_jobs=8,verbosity=-1,
            deterministic=True,force_col_wise=True,lambdarank_truncation_level=55,label_gain=[0,1])
        print('Fit final query ranker',x.shape,flush=True)
        model.fit(x,data['y'],group=data['sizes'],sample_weight=np.repeat(data['group_weight'],data['sizes']))
        save_cache(model,path)
        return model,path
    finally:
        training,training_history,QR_CACHE,QR_FP=old_frame,old_history,old_cache,old_fp
        globals()['training_documents']=original_function


def export_query_candidate():
    selection=json.loads((QR_CACHE/'selection.json').read_text(encoding='utf-8'))
    winner=selection['winner']
    assert winner['variant']!='v5','No new candidate selected on development'
    ids,documents=full_document_vectors()
    model_path,vector_path=fit_final_encoder(ids,documents)
    ranker_path=None
    if winner['variant']=='query_ranker':
        model,ranker_path=fit_final_query_ranker(ids,documents,int(winner['trees']))
    vectors=np.load(vector_path,allow_pickle=False)
    lookup={text:i for i,text in enumerate(sorted(set(queries.query_norm)))}
    pool_path=V5_CACHE/f'benchmark_{V5_FP}.joblib'
    # The v5 cache may use a different name; use the same final history if absent.
    if pool_path.exists():
        records,features=joblib.load(pool_path)
    else:
        history=V5History(history_all)
        records=retrieve_semantic_features(queries,history,progress_every=500)
        features=[v5_features(query,record,history) for query,record in zip(queries.itertuples(index=False),records)]
        del history
    original=baseline_v5(records,features,final=True)
    added=[query_features(vectors[lookup[text]],item_rows,raw)
        for text,(item_rows,raw) in zip(queries.query_norm,records)]
    if winner['variant']=='direct_query_blend':
        scores=blend_scores([x[:,1] for x in added],original,float(winner['weight']))
    else:
        scores=blend_scores(predict_scores(model,[np.column_stack([x,y]) for x,y in zip(features,added)],int(winner['trees'])),original,float(winner['weight']))
    predictions=top50(records,scores)
    result=pd.DataFrame({'query_id':queries.query_id,'answer':[' '.join(ITEM_IDS[row] for row in prediction) for prediction in predictions]})
    assert len(result)==2452 and result.query_id.is_unique and set(result.query_id)==set(queries.query_id)
    assert all(len(p)==50 and len(set(p))==50 for p in predictions)
    output=ROOT/'answer_query_encoder.csv'
    result.to_csv(output,index=False,encoding='utf-8',lineterminator='\n')
    manifest={'fingerprint':EXPORT_FP,'query_pilot_fingerprint':QE_FP,'query_rank_fingerprint':selection['fingerprint'],
        'selection':selection,'control':json.loads((ROOT/'artifacts/query-encoder-rank/control.json').read_text()),
        'answer_file':output.name,'answer_sha256':sha256_file(output),'input_sha256':input_hashes,
        'encoder':model_path.relative_to(ROOT).as_posix(),'query_vectors':vector_path.relative_to(ROOT).as_posix(),
        'encoder_sha256':sha256_file(model_path/'model.safetensors'),'query_vectors_sha256':sha256_file(vector_path),
        'final_ranker':ranker_path.relative_to(ROOT).as_posix() if ranker_path else None,
        'final_ranker_sha256':sha256_file(ranker_path) if ranker_path else None,
        'document_encoder_frozen':True,'training_pairs':len(history_all),'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH}
    assert manifest['main_answer_unchanged']
    (EXPORT_CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('QUERY CANDIDATE',json.dumps(manifest),flush=True)


if __name__=='__main__':
    if '--prepare-encoder' in __import__('sys').argv:
        ids,documents=full_document_vectors()
        model_path,vector_path=fit_final_encoder(ids,documents)
        print('Full-train query encoder prepared:',model_path,vector_path,flush=True)
    else:
        export_query_candidate()
