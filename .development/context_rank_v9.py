"""Bounded expanded-pool OOF training with independent context features.

The first comparison uses a fixed 4,000-context subset of the existing training
split, both history regimes and all six cached OOF query encoders. Sampling keeps
more competitors than v7 and mines the frozen training ranker. This miner is not
claimed to be OOF; history/encoder feature producers are OOF. Development chooses
features/blend before the reused control is opened. Confirmed answers are intact.
"""
from pathlib import Path
import ast
import gc
import hashlib
import json
import time

CONTEXT_DRIVER=Path(__file__).resolve()
source=CONTEXT_DRIVER.with_name('expanded_pool_rank_v8.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
DRIVER=CONTEXT_DRIVER
__file__=str(DRIVER)
from context_signals_v9 import GeographicEvidence,SemanticEvidence,GEO_NAMES,NEIGHBOR_NAMES

CONTEXT_CACHE=ROOT/'artifacts/context-v9'
CONTEXT_CACHE.mkdir(exist_ok=True)
CONTEXT_CONFIG={'evaluation_contexts':4000,'final_contexts':12000,'chunk':128,
    'mined_negatives':500,'teacher_negatives':160,'learned_negatives':160,'random_tail':96,
    'neighbors':24,'trees':400,'seed':SEED+1201,'category_input':False}
CONTEXT_FP=hashlib.sha256(json.dumps({'base':QR_FP,'inputs':input_hashes,'config':CONTEXT_CONFIG,
    'module':sha256_file(DRIVER.with_name('context_signals_v9.py')),'driver':sha256_file(DRIVER)},sort_keys=True).encode()).hexdigest()[:16]
FEATURE_NAMES=V5_FEATURE_NAMES+QR_FEATURE_NAMES+GEO_NAMES+NEIGHBOR_NAMES
MODULE=__import__('sys').modules['__main__']
module_globals=globals()

def prepare_resources():
    # Freeze metadata only. Labels always arrive through explicitly isolated history.
    coordinates=pd.read_parquet(ROOT/'train.parquet',columns=['item_id','item_latitude','item_longitude'])
    coordinates=coordinates.drop_duplicates('item_id').set_index('item_id').rename(
        columns={'item_latitude':'lat','item_longitude':'lon'})
    coordinates=coordinates.apply(pd.to_numeric,errors='coerce').astype(np.float64)
    invalid=(~np.isfinite(coordinates).all(axis=1))|(coordinates.lat.abs()>90)|(coordinates.lon.abs()>180)|((coordinates.lat==0)&(coordinates.lon==0))
    coordinates.loc[invalid,['lat','lon']]=np.nan
    full_manifest=json.loads((ROOT/'artifacts/query-encoder-candidate/manifest.json').read_text())
    ids=sorted(history_all.item_id.unique())
    documents=np.load(ROOT/f'artifacts/query-encoder-candidate/documents_{full_manifest["fingerprint"]}.npy',mmap_mode='r',allow_pickle=False)
    assert documents.shape==(len(ids),384)
    space=prepare_micro_space()
    lookup={}
    for text,vector in zip(space['inputs']['query'],space['dense'][:,:384]):lookup.setdefault(text,vector.copy())
    del space;gc.collect()
    return coordinates,ids,documents,lookup

RESOURCES=None
GEO_EVIDENCE=None

def install_retriever():
    # This extends our own frozen v8 function, never another candidate's source.
    source=install_expanded_retriever()
    source=source.replace('def retrieve_expanded_features(', 'def retrieve_context_features(')
    source=source.replace('semantic_candidates = np.union1d(semantic_candidates, learned_candidates)',
        '''extra_geo = GEO_EVIDENCE.additional_candidates(loc, cosine, learned_cosine,
            items.item_latitude.to_numpy(), items.item_longitude.to_numpy(), stable_topk)
        semantic_candidates = np.union1d(semantic_candidates, np.union1d(learned_candidates, extra_geo))''')
    exec(compile(source,str(DRIVER),'exec'),globals())
    return source

def evidence(history,frame):
    global GEO_EVIDENCE
    coordinates,ids,documents,lookup=RESOURCES
    assert not set(frame.context_key)&set(history.context_key)
    GEO_EVIDENCE=GeographicEvidence(history,coordinates,{k:v for k,v in centers.iterrows()})
    neighbors=SemanticEvidence(history,lookup,ids,documents,MICROS)
    teacher=np.stack([lookup[text] for text in frame.query_norm])
    results=neighbors.query_evidence(teacher,CONTEXT_CONFIG['neighbors'])
    del neighbors;gc.collect()
    return results

def matrices(frame,records,features,neighbor_results):
    prototypes,probabilities,statistics=neighbor_results
    output=[]
    lat=items.item_latitude.to_numpy();lon=items.item_longitude.to_numpy()
    for i,(query,(ids,raw),base) in enumerate(zip(frame.itertuples(index=False),records,features)):
        geography=GEO_EVIDENCE.features(query.search_location_id,lat[ids],lon[ids])
        neighbor=SemanticEvidence.features(prototypes[i],probabilities[i],statistics[i],
            semantic_index.items[ids],ITEM_MICRO_COLS[ids],rankdata)
        x=np.column_stack([base,query_features(LEARNED_LOOKUP[query.query_norm],ids,raw),geography,neighbor])
        assert x.shape[1]==len(FEATURE_NAMES)==70 and not np.isinf(x).any()
        output.append(x.astype(np.float32))
    return output

def selected_training(frame,count):
    # Stable context sampling, unrelated to test IDs or labels. Prefixes permit
    # increasing the sample without replacing all previously prepared groups.
    ordered=frame.context_key.map(lambda key:hashlib.sha256(f'{CONTEXT_CONFIG["seed"]}:{key}'.encode()).hexdigest())
    return set(frame.loc[ordered.sort_values().index[:count],'context_key'])

def prepare_training(stage='evaluation'):
    global LEARNED_LOOKUP
    output=CONTEXT_CACHE/f'{stage}_training_{CONTEXT_FP}.joblib'
    if output.exists():return joblib.load(output)
    if stage=='evaluation':frame=training;source_history=training_history;encoder_cache=QR_CACHE;encoder_fp=QR_FP
    else:
        frame=history_all[history_all.context_key.isin(gold)].drop_duplicates('context_key')
        frame=frame[[*QUERY_COLS,'query_norm','context_key']].sort_values('context_key').reset_index(drop=True)
        source_history=history_all;encoder_cache=ROOT/'artifacts/query-encoder-candidate'
        encoder_fp=json.loads((encoder_cache/'manifest.json').read_text())['fingerprint']
    picked=selected_training(frame,CONTEXT_CONFIG[f'{stage}_contexts'])
    miner=joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib') if stage=='evaluation' else joblib.load(
        ROOT/json.loads((ROOT/'artifacts/query-encoder-candidate/manifest.json').read_text())['final_ranker'])
    truth=labels_from_gold(gold,frame)
    xparts=[];yparts=[];sizes=[];known=[];audits=[]
    random=np.random.default_rng(CONTEXT_CONFIG['seed'])
    for mode in ['unseen_text','held_context']:
        assignments=oof_assignments(frame,mode,3,SEED+803)
        for fold in range(3):
            positions=np.flatnonzero(assignments==fold)
            fold_frame=frame.iloc[positions];fold_truth=[truth[i] for i in positions]
            history,_,_=history_for_queries(source_history,fold_frame,fold_truth,ITEM_IDS,mode,.9,SEED+804+fold)
            bundle=joblib.load(encoder_cache/f'{mode}_{fold}_queries_{encoder_fp}.joblib')
            assert bundle['history_sha256']==history_digest(history)
            LEARNED_LOOKUP=dict(zip(bundle['texts'],bundle['vectors']))
            queries_subset=fold_frame[fold_frame.context_key.isin(picked)]
            audit=history_coverage(queries_subset,labels_from_gold(gold,queries_subset),history,ITEM_IDS)
            assert audit['own_context_overlap']==0
            audit.update(mode=mode,fold=fold,encoder_history_sha256=bundle['history_sha256'])
            audits.append(audit)
            neighbors=evidence(history,queries_subset)
            retriever_history=V5History(history)
            del history,bundle;gc.collect()
            print('Context OOF',stage,mode,fold,len(queries_subset),flush=True)
            for start in range(0,len(queries_subset),CONTEXT_CONFIG['chunk']):
                chunk=queries_subset.iloc[start:start+CONTEXT_CONFIG['chunk']]
                chunk_path=CONTEXT_CACHE/f'{stage}_{mode}_{fold}_{start}_{CONTEXT_FP}.joblib'
                if chunk_path.exists():
                    saved=joblib.load(chunk_path)
                else:
                    records=retrieve_context_features(chunk,retriever_history,progress_every=100000)
                    original=[v5_features(q,r,retriever_history) for q,r in zip(chunk.itertuples(index=False),records)]
                    all_features=matrices(chunk,records,original,tuple(n[start:start+len(chunk)] for n in neighbors))
                    saved=[]
                    for q,(ids,raw),x in zip(chunk.itertuples(index=False),records,all_features):
                        target=np.isin(ids,list(gold[q.context_key]))
                        positive=np.flatnonzero(target);negative=np.flatnonzero(~target)
                        if not len(positive) or not len(negative):continue
                        scores=miner.predict(x[negative,:51],num_iteration=400)
                        hard=np.unique(np.concatenate([negative[stable_topk(scores,CONTEXT_CONFIG['mined_negatives'])],
                            negative[stable_topk(raw[negative,7],CONTEXT_CONFIG['teacher_negatives'])],
                            negative[stable_topk(x[negative,51-6],CONTEXT_CONFIG['learned_negatives'])]]))
                        tail=np.setdiff1d(negative,hard)
                        # A context-derived seed makes chunk cache resumption invariant.
                        seed=int(hashlib.sha256(f'{stage}:{mode}:{q.context_key}'.encode()).hexdigest()[:8],16)
                        rng=np.random.default_rng(seed)
                        random_tail=rng.choice(tail,min(len(tail),CONTEXT_CONFIG['random_tail']),replace=False)
                        selected=np.sort(np.concatenate([positive,hard,random_tail]))
                        assert not set(ids[selected[~target[selected]]])&gold[q.context_key]
                        saved.append((x[selected],target[selected].astype(np.uint8),q.query_norm in retriever_history.text_to_row))
                    save_cache(saved,chunk_path)
                    del records,original,all_features
                for x,y,is_known in saved:
                    xparts.append(x);yparts.append(y);sizes.append(len(y));known.append(is_known)
                if start%512==0:print('Prepared context groups',len(sizes),'rows',sum(sizes),flush=True)
            del retriever_history,neighbors;gc.collect()
    known=np.asarray(known,bool);observed=known.mean()
    group_weight=np.where(known,known_target/max(observed,1e-6),(1-known_target)/max(1-observed,1e-6)).astype(np.float32)
    data={'X':np.concatenate(xparts),'y':np.concatenate(yparts),'sizes':np.asarray(sizes,np.int32),
        'known':known,'group_weight':group_weight,'audits':audits}
    save_cache(data,output)
    report={'stage':stage,'sampled_contexts':len(picked),'groups':len(sizes),'rows':len(data['y']),
        'mean_group_size':float(np.mean(sizes)),'positive_rows':int(data['y'].sum()),'audits':audits,
        'feature_names':FEATURE_NAMES,'mining_model':'frozen v7 trained ranker; not OOF miner',
        'history_and_query_features_oof':True}
    (output.with_suffix('.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    return data

FEATURE_SETS={'base':list(range(51)),'geography':list(range(61)),
              'neighbors':list(range(51))+list(range(61,70)),'combined':list(range(70))}

def fit_models(data,stage='evaluation',variants=None):
    models={}
    for name in variants or FEATURE_SETS:
        path=CONTEXT_CACHE/f'{stage}_{name}_{CONTEXT_FP}.joblib'
        if path.exists():models[name]=joblib.load(path);continue
        columns=FEATURE_SETS[name]
        x=np.ascontiguousarray(data['X'][:,columns])
        model=lgb.LGBMRanker(n_estimators=400,num_leaves=31,learning_rate=.05,reg_lambda=10,
            max_bin=127,min_child_samples=50,random_state=SEED,n_jobs=8,verbosity=-1,
            deterministic=True,force_col_wise=True,lambdarank_truncation_level=55,label_gain=[0,1])
        started=time.perf_counter();print('Fit context',stage,name,x.shape,flush=True)
        model.fit(x,data['y'],group=data['sizes'],sample_weight=np.repeat(data['group_weight'],data['sizes']))
        save_cache(model,path);models[name]=model
        (path.with_suffix('.json')).write_text(json.dumps({'seconds':round(time.perf_counter()-started,2),
            'columns':columns,'feature_names':[FEATURE_NAMES[c] for c in columns],'sha256':sha256_file(path)},indent=2),encoding='utf-8')
        del x;gc.collect()
    return models

def evaluate(frame,mode,stage,models,other):
    global LEARNED_LOOKUP
    path=CONTEXT_CACHE/f'{stage}_{mode}_pools_{CONTEXT_FP}.joblib'
    history=evaluation_history(frame,mode,other)
    known=frame.query_norm.isin(history.query_norm).to_numpy()
    audit=history_coverage(frame,labels_from_gold(gold,frame),history,ITEM_IDS)
    vectors=evaluation_query_features(frame,'development' if stage=='development' else 'control')
    LEARNED_LOOKUP=dict(zip(sorted(set(frame.query_norm)),vectors))
    if path.exists():records,features,all_features=joblib.load(path)
    else:
        neighbors=evidence(history,frame)
        retriever_history=V5History(history)
        records=retrieve_context_features(frame,retriever_history,progress_every=1000)
        features=[v5_features(q,r,retriever_history) for q,r in zip(frame.itertuples(index=False),records)]
        all_features=matrices(frame,records,features,neighbors)
        save_cache((records,features,all_features),path)
        del neighbors,retriever_history
    del history;gc.collect()
    baseline=baseline_v5(records,features)
    reference=joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    v7_scores=blend_scores(predict_scores(reference,[x[:,:51] for x in all_features],400),baseline,.75)
    truth=labels_from_gold(gold,frame)
    values={'v7_context_pool':per_query_recall(top50(records,v7_scores),truth)}
    for name,model in models.items():
        selected=[np.ascontiguousarray(x[:,FEATURE_SETS[name]]) for x in all_features]
        score=predict_scores(model,selected,400)
        for weight in [.5,.75,1.]:
            values[(name,weight)]=per_query_recall(top50(records,blend_scores(score,v7_scores,weight)),truth)
        del selected,score;gc.collect()
    pool_recall=float(per_query_recall([r[0] for r in records],truth).mean())
    del records,features,all_features,baseline,v7_scores;gc.collect()
    return values,known,audit,pool_recall

def comparison():
    global RESOURCES
    install_retriever();RESOURCES=prepare_resources()
    data=prepare_training()
    models=fit_models(data)
    del data;gc.collect()
    unseen,_,audit_a,pool_a=evaluate(development,'unseen_text','development',models,control)
    held,known,audit_b,pool_b=evaluate(development,'held_context','development',models,control)
    query_known=queries.query_norm.isin(history_all.query_norm)
    rows=[]
    for key in unseen:
        a=matched_slice_recall(development,unseen[key],queries[~query_known])
        b=matched_slice_recall(development,held[key],queries[query_known],known)
        name,weight=key if isinstance(key,tuple) else (key,0.)
        rows.append({'variant':name,'weight':weight,'matched_recall50':(1-known_target)*a+known_target*b,
            'unseen_macro':float(unseen[key].mean()),'held_macro':float(held[key].mean())})
    rows.append({'variant':'confirmed_v8_pool','weight':0.,'matched_recall50':.9528152912874686,
        'unseen_macro':.9478676470588235,'held_macro':.9490441176470589})
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(CONTEXT_CACHE/'development.csv',index=False)
    eligible=table[table.unseen_macro>=.9478676470588235-.002]
    winner=eligible.iloc[0].to_dict()
    selection={'fingerprint':CONTEXT_FP,'winner':winner,'control_used_for_selection':False,
        'pool_recall':[pool_a,pool_b],'audits':[audit_a,audit_b],
        'limitations':['Reused development/control. Frozen v4 auxiliary-prior limitation.',
            'A 4000-context training pilot, not all-train ranker refit.',
            'Frozen v7 miner is trained on training labels; not an OOF miner.'],
        'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH}
    assert selection['main_answer_unchanged']
    (CONTEXT_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Context selection',json.dumps(selection),flush=True)
    # Only the frozen development winner is evaluated on control.
    chosen={winner['variant']:models[winner['variant']]} if winner['variant'] in models else {}
    result,_,audit,pool=evaluate(control,'unseen_text','control',chosen,development)
    if winner['variant']=='confirmed_v8_pool':score=.9533333333333334
    else:
        key=(winner['variant'],winner['weight']) if winner['variant'] in chosen else 'v7_context_pool'
        score=float(result[key].mean())
    report={'selected_recall50':score,'v8_recall50':.9533333333333334,'pool_recall':pool,
        'history_audit':audit,'previously_viewed_control':True,'used_for_selection':False}
    (CONTEXT_CACHE/'control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Context control',json.dumps(report),flush=True)

if __name__=='__main__':comparison()
