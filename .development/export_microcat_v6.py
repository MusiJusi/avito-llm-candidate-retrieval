"""Refit development-selected microcat ranker; export a validated new candidate.

Control metrics are reported without changing the selected settings. A v5 win on
development preserves the existing submission. Inference can load final weights
without regenerating the large OOF training groups.
"""
from pathlib import Path
import ast
import json

EXPORT_DRIVER=Path(__file__).resolve()
rank_source=EXPORT_DRIVER.with_name('microcat_rank_v6.py')
tree=ast.parse(rank_source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(rank_source)
exec(compile(tree,str(rank_source),'exec'),globals())
__file__=str(EXPORT_DRIVER)


def compress_final_classifier(component,path):
    """Storage compression preserves numerical arrays and fits GitHub file limits."""
    with path.open('rb') as stream:
        raw_pickle=stream.read(1)==b'\x80'
    if raw_pickle and path.stat().st_size>100*2**20:
        temporary=path.with_suffix('.joblib.tmp')
        joblib.dump(component,temporary,compress=3)
        temporary.replace(path)


def export_microcat(selection, control_result):
    winner=selection['winner']
    assert selection['rank_fingerprint']==RANK_FP and selection['micro_fingerprint']==MICRO_FP
    if winner['variant']=='v5':
        print('Microcat did not improve development. Original answer preserved.',flush=True)
        return
    space=prepare_micro_space()
    frame=history_all[history_all.context_key.isin(gold)].drop_duplicates('context_key')
    frame=frame[[*QUERY_COLS,'query_norm','context_key']].sort_values('context_key').reset_index(drop=True)
    variant=winner['variant']
    model_path=MICRO_CACHE/f'final_rank_{variant}_{RANK_FP}.joblib'
    if USE_CACHE and model_path.exists():
        model=joblib.load(model_path)
    else:
        if not (V5_CACHE/f'final_mined_{V5_FP}.joblib').exists():
            pilot=joblib.load(V5_CACHE/f'mixed_wide_{V5_FP}.joblib')
            old_data=prepare_v5_training(frame,history_all,'final',miner=pilot)
            del old_data,pilot
            gc.collect()
        data,extra=training_micro_features(space,frame,history_all,'final')
        model=fit_micro_ranker(data,extra,variant,'final',int(winner['trees']))
        del data,extra
        gc.collect()
    kinds=['nb','mlp'] if variant=='both' else [variant]
    predictions={}
    model_paths={}
    for kind in kinds:
        component=classifier(space,history_all,'final_all',kind)
        compress_final_classifier(component,MICRO_CACHE/f'final_all_{kind}_{MICRO_FP}.joblib')
        predictions[kind]={'probabilities':predict_micro(component,space,queries),'supported':component['supported']}
        model_paths[kind]=f'final_all_{kind}_{MICRO_FP}.joblib'
        del component
    records,features=joblib.load(V5_CACHE/f'benchmark_{V5_FP}.joblib') if (V5_CACHE/f'benchmark_{V5_FP}.joblib').exists() else (None,None)
    if records is None:
        history=V5History(history_all)
        records=retrieve_semantic_features(queries,history,progress_every=500)
        features=[v5_features(q,r,history) for q,r in zip(queries.itertuples(index=False),records)]
        del history
    augmented=[]
    for position,((ids,_),base) in enumerate(zip(records,features)):
        parts=[micro_candidate_features(predictions[kind]['probabilities'][position],ITEM_MICROCATS[ids],
            space['classes'],predictions[kind]['supported']) for kind in kinds]
        augmented.append(np.column_stack([base,*parts]))
    new_scores=predict_scores(model,augmented,int(winner['trees']))
    old_scores=baseline_v5(records,features,final=True)
    chosen=blend_scores(new_scores,old_scores,float(winner['weight']))
    ids=top50(records,chosen)
    answer=pd.DataFrame({'query_id':queries.query_id.astype(str),'answer':[' '.join(ITEM_IDS[row]) for row in ids]})
    assert list(answer.columns)==['query_id','answer'] and len(answer)==len(queries)
    assert answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    allowed=set(ITEM_IDS)
    for q,text in answer.itertuples(index=False,name=None):
        values=text.split(' ')
        assert len(q)==16 and len(values)==50 and len(set(values))==50
        assert all(value in allowed and re.fullmatch('[0-9a-f]{16}',value) for value in values)
    repeated=top50(records,blend_scores(predict_scores(model,augmented,int(winner['trees'])),old_scores,float(winner['weight'])))
    assert all(np.array_equal(a,b) for a,b in zip(ids,repeated))
    backup=ROOT/'answer_v0.5.csv'
    if not backup.exists():
        assert sha256_file(ROOT/'answer.csv')==V5_MANIFEST['answer_sha256']
        backup.write_bytes((ROOT/'answer.csv').read_bytes())
    answer.to_csv(MICRO_CACHE/'answer_candidate.csv',index=False,encoding='utf-8',lineterminator='\n')
    assert pd.read_csv(MICRO_CACHE/'answer_candidate.csv',dtype=str,keep_default_na=False).equals(answer)
    # Mixed evidence on held control: keep the existing main submission intact.
    # A separate reproducible candidate is useful for later model combinations.
    candidate_file=ROOT/'answer_microcat_v6.csv'
    candidate_file.write_bytes((MICRO_CACHE/'answer_candidate.csv').read_bytes())
    report={'stage':'microcat-v6','selection':selection,'control':control_result,
        'micro_fingerprint':MICRO_FP,'rank_fingerprint':RANK_FP,'input_sha256':input_hashes,
        'answer_file':candidate_file.name,'answer_sha256':sha256_file(candidate_file),'previous_answer_sha256':sha256_file(backup),
        'final_ranker':model_path.name,'final_classifiers':model_paths,
        'input_vectors':f'input_vectors_{MICRO_FP}.npy','final_contexts':len(frame),
        'classifier_training_pairs':len(history_all),'feature_count':45+7*len(kinds),
        'frozen_v5_fingerprint':V5_FP,'encoder_finetuned':False,'external_api_used':False,
        'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==V5_MANIFEST['answer_sha256'],
        'status':'experimental candidate; development improved, held-control evidence mixed'}
    (MICRO_CACHE/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('MICRO FINAL',json.dumps(report),flush=True)


if __name__=='__main__':
    export_microcat(json.loads((MICRO_CACHE/'ranker_selection.json').read_text()),
                    json.loads((MICRO_CACHE/'ranker_control.json').read_text()))
