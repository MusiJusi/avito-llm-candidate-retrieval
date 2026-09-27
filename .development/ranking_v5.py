"""Train v5 locally: complete gold, mixed OOF histories, wider hard negatives.

Development helpers initialize existing unsupervised indexes. Numerical functions
are embedded in the delivered notebook after selection. No benchmark labels or
network API are used. The existing answer is replaced only after final validation.
"""
from pathlib import Path
import sys, json, hashlib, time, gc, os

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding='utf-8')
sys.path[:0] = [str(ROOT / folder) for folder in ['.ranking_deps', '.inspection_deps', '.semantic_deps']]
from bootstrap_ranking import bootstrap
from validation_protocol import freeze_gold, labels_from_gold, history_for_queries, oof_assignments, history_coverage
bootstrap(globals())
import lightgbm as lgb
from scipy.stats import rankdata

INDEX_CACHE = CACHE
OLD_CACHE = ROOT / 'artifacts/ranking-v1'
CACHE = ROOT / 'artifacts/ranking-v5'
CACHE.mkdir(exist_ok=True)
V5_CONFIG = {'data_revision': 1, 'chunk': 128, 'folds': 3, 'cold_fraction': .9,
    'hard_lexical': 100, 'hard_geo_semantic': 100, 'hard_semantic': 60,
    'random_tail': 64, 'mined_hard': 160, 'trees': 400}
fingerprint = hashlib.sha256(json.dumps({'inputs': input_hashes, 'config': V5_CONFIG,
    'protocol': hashlib.sha256((ROOT/'.development/validation_protocol.py').read_bytes()).hexdigest(),
    'pipeline': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()[:16]
gold = freeze_gold(history_all, ITEM_TO_ROW)
PROTOCOL_CACHE = ROOT / 'artifacts/validation-v5'
training = pd.read_parquet(PROTOCOL_CACHE/'training_contexts.parquet').drop(columns='relevant_item_ids')
development = pd.read_parquet(PROTOCOL_CACHE/'development_contexts.parquet').drop(columns='relevant_item_ids')
control = pd.read_parquet(PROTOCOL_CACHE/'held_control_contexts.parquet').drop(columns='relevant_item_ids')
all_evaluation = pd.concat([development, control], ignore_index=True)
all_evaluation_truths = labels_from_gold(gold, all_evaluation)
training_history, evaluation_cold, _ = history_for_queries(history_all, all_evaluation,
    all_evaluation_truths, ITEM_IDS, 'unseen_text', .9, SEED+802)
assert not set(training.query_norm) & set(all_evaluation.query_norm)
assert all(labels_from_gold(gold, training))
known_target = float(queries.query_norm.isin(history_all.query_norm).mean())

# Reuse content-addressed document vectors; encode only this run's query set.
for file in INDEX_CACHE.glob('e5_items_*.npy'):
    target = CACHE/file.name
    if not target.exists():
        os.link(file, target)
semantic_index.prepare(pd.concat([training.query_norm, development.query_norm,
    control.query_norm, queries.query_norm, history_all.loc[history_all.item_id.isin(ITEM_TO_ROW),'query_norm']]))
EXTRA_FEATURE_NAMES = ['history_query_known', 'history_query_log_pairs',
    'history_query_micro_entropy', 'history_query_top_micro_probability',
    'geo_center_available', 'geo_transition_probability', 'geo_transition_log_support']
V5_FEATURE_NAMES = list(RANK_FEATURE_NAMES) + EXTRA_FEATURE_NAMES


class V5History(HistorySignals):
    """Base retrieval statistics plus explicit evidence strength for ranking."""
    def __init__(self, history):
        super().__init__(history)
        self.query_counts = history.groupby('query_norm').size().to_dict()
        matrix = self.micro_matrix.tocsr()
        self.entropy = np.zeros(len(self.texts), dtype=np.float32)
        self.peak = np.zeros(len(self.texts), dtype=np.float32)
        for row in range(len(self.texts)):
            probabilities = matrix.data[matrix.indptr[row]:matrix.indptr[row+1]]
            if len(probabilities):
                self.entropy[row] = -(probabilities*np.log(np.maximum(probabilities,1e-12))).sum()/np.log(max(len(MICROS),2))
                self.peak[row] = probabilities.max()
        self.location_counts = history.groupby('search_location_id').size().to_dict()
        counts = history.groupby(['search_location_id','item_location_id']).size()
        self.location_probability = {}
        for loc, group in counts.groupby(level=0):
            total = float(group.sum())
            self.location_probability[int(loc)] = {int(pair[1]): float(count/total) for pair,count in group.items()}


def v5_features(query, record, history):
    ids, _ = record
    original = rank_semantic_features(query, record)
    row = history.text_to_row.get(query.query_norm)
    known = row is not None
    center = query.search_location_id in centers.index and np.isfinite(centers.loc[query.search_location_id].to_numpy(dtype=float)).all()
    constants = [float(known), np.log1p(history.query_counts.get(query.query_norm,0)),
        float(history.entropy[row]) if known else np.nan,
        float(history.peak[row]) if known else np.nan, float(center)]
    mapping = history.location_probability.get(int(query.search_location_id), {})
    probability = np.array([mapping.get(int(loc),0) for loc in ITEM_LOCS[ids]], dtype=np.float32)
    support = np.full(len(ids),np.log1p(history.location_counts.get(query.search_location_id,0)),dtype=np.float32)
    output = np.column_stack([original,np.tile(constants,(len(ids),1)),probability,support]).astype(np.float32)
    assert output.shape[1] == len(V5_FEATURE_NAMES) and not np.isinf(output).any()
    return output


def sample_v5_group(ids, base, features, truth, random, miner=None):
    target = np.isin(ids,list(truth))
    positive, negative = np.flatnonzero(target), np.flatnonzero(~target)
    if not len(positive) or not len(negative):
        return np.array([],dtype=np.int32), target
    pieces = [negative[stable_topk(score_candidates(base[negative],best_config),V5_CONFIG['hard_lexical'])],
        negative[stable_topk(base[negative,7],V5_CONFIG['hard_geo_semantic'])],
        negative[stable_topk(base[negative,6],V5_CONFIG['hard_semantic'])]]
    if miner is not None:
        scores = miner.predict(features[negative],num_iteration=200)
        pieces.append(negative[stable_topk(scores,V5_CONFIG['mined_hard'])])
    hard = np.unique(np.concatenate(pieces))
    tail = np.setdiff1d(negative,hard)
    sampled = random.choice(tail,min(len(tail),V5_CONFIG['random_tail']),replace=False)
    selected = np.sort(np.concatenate([positive,hard,sampled])).astype(np.int32)
    assert not set(ids[selected[~target[selected]]]) & truth
    return selected,target


def prepare_v5_training(qframe, source_history, stage, miner=None):
    """Reuse full-pool chunks for mining; never inject a missing positive item."""
    tag = 'mined' if miner is not None else 'initial'
    output_path = CACHE/f'{stage}_{tag}_{fingerprint}.joblib'
    if USE_CACHE and output_path.exists():
        return joblib.load(output_path)
    truths = labels_from_gold(gold,qframe)
    xs,ys,sizes,known_flags,modes,query_positions,audits = [],[],[],[],[],[],[]
    random = np.random.default_rng(SEED+811)
    for mode in ['unseen_text','held_context']:
        assignments = oof_assignments(qframe,mode,3,SEED+803)
        for fold in range(3):
            positions = np.flatnonzero(assignments==fold)
            fold_queries = qframe.iloc[positions]
            fold_truths = [truths[i] for i in positions]
            fit,cold,actual_known = history_for_queries(source_history,fold_queries,fold_truths,
                ITEM_IDS,mode,.9,SEED+804+fold)
            audit = history_coverage(fold_queries,fold_truths,fit,ITEM_IDS)
            audit.update(mode=mode,fold=fold,cold_items=len(cold))
            assert audit['own_context_overlap']==0
            audits.append(audit)
            history = None
            print(stage,tag,mode,'fold',fold,'contexts',len(positions),flush=True)
            for start in range(0,len(positions),V5_CONFIG['chunk']):
                path = CACHE/f'{stage}_pool_{mode}_{fold}_{start}_{fingerprint}.joblib'
                chunk = fold_queries.iloc[start:start+V5_CONFIG['chunk']]
                if USE_CACHE and path.exists():
                    records,matrices = joblib.load(path)
                else:
                    if history is None:
                        history = V5History(fit)
                    records = retrieve_semantic_features(chunk,history,progress_every=100000)
                    matrices = [v5_features(q,r,history) for q,r in zip(chunk.itertuples(index=False),records)]
                    save_cache((records,matrices),path)
                for offset,((ids,base),features) in enumerate(zip(records,matrices)):
                    selected,target = sample_v5_group(ids,base,features,fold_truths[start+offset],random,miner)
                    if not len(selected):
                        continue
                    xs.append(features[selected]); ys.append(target[selected].astype(np.uint8))
                    sizes.append(len(selected)); known_flags.append(bool(actual_known[start+offset]))
                    modes.append(mode); query_positions.append(int(positions[start+offset]))
                if start%1024==0:
                    print(stage,mode,fold,'processed',start+len(chunk),flush=True)
            del history,fit,records,matrices
            gc.collect()
    known = np.array(known_flags,dtype=bool)
    observed = float(known.mean())
    group_weights = np.where(known,known_target/max(observed,1e-6),
        (1-known_target)/max(1-observed,1e-6)).astype(np.float32)
    data = {'X':np.concatenate(xs),'y':np.concatenate(ys),'sizes':np.array(sizes,dtype=np.int32),
        'known':known,'mode':np.array(modes),'query_position':np.array(query_positions,dtype=np.int32),
        'group_weight':group_weights,'audit':audits}
    save_cache(data,output_path)
    report = {'contexts':len(qframe),'groups':len(sizes),'rows':len(data['y']),
        'positives':int(data['y'].sum()),'mean_group_size':float(np.mean(sizes)),
        'actual_known_fraction':observed,'target_known_fraction':known_target,'oof_audits':audits}
    (CACHE/f'{stage}_{tag}_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Prepared',json.dumps({k:v for k,v in report.items() if k!='oof_audits'}),flush=True)
    return data


def fit_v5(data,name,mode=None,trees=400):
    path = CACHE/f'{name}_{fingerprint}.joblib'
    if USE_CACHE and path.exists():
        return joblib.load(path)
    group_mask = np.ones(len(data['sizes']),dtype=bool) if mode is None else data['mode']==mode
    row_mask = np.repeat(group_mask,data['sizes'])
    x,y = data['X'][row_mask],data['y'][row_mask]
    sizes = data['sizes'][group_mask]
    weights = np.repeat(data['group_weight'][group_mask],sizes)
    if mode=='unseen_text':
        weights[:] = 1
    model = lgb.LGBMRanker(n_estimators=trees,num_leaves=31,learning_rate=.05,
        max_bin=127,min_child_samples=50,reg_lambda=10,random_state=SEED,n_jobs=8,
        verbosity=-1,deterministic=True,force_col_wise=True,
        lambdarank_truncation_level=55,label_gain=[0,1])
    started = time.perf_counter()
    print('Fit',name,'rows',len(y),'groups',len(sizes),flush=True)
    model.fit(x,y,group=sizes,sample_weight=weights)
    save_cache(model,path)
    print('Fitted',name,'seconds',round(time.perf_counter()-started,1),flush=True)
    return model


def evaluation_features(frame,mode,stage,other_frame):
    path = CACHE/f'{stage}_{mode}_{fingerprint}.joblib'
    if USE_CACHE and path.exists():
        return joblib.load(path)
    truths = labels_from_gold(gold,frame)
    other_truths = labels_from_gold(gold,other_frame)
    other_positive = np.array(sorted({ITEM_IDS[i] for truth in other_truths for i in truth}),dtype=str)
    random = np.random.default_rng(SEED+812)
    other_cold = set(random.choice(other_positive,int(.9*len(other_positive)),replace=False))
    fit,cold,known = history_for_queries(history_all,frame,truths,ITEM_IDS,mode,.9,SEED+801,
        excluded_texts=other_frame.query_norm,excluded_items=other_cold)
    history = V5History(fit)
    records = retrieve_semantic_features(frame,history,progress_every=500)
    matrices = [v5_features(q,r,history) for q,r in zip(frame.itertuples(index=False),records)]
    output = (records,matrices,known,history_coverage(frame,truths,fit,ITEM_IDS))
    save_cache(output,path)
    del history,fit
    gc.collect()
    return output


def predict_scores(model,matrices,trees=200):
    output=[]
    for start in range(0,len(matrices),64):
        block=matrices[start:start+64]
        values=model.predict(np.concatenate(block),num_iteration=trees)
        output.extend(np.split(values,np.cumsum([len(x) for x in block])[:-1]))
    return output


def top50(records,scores):
    return [ids[stable_topk(score,50)] for (ids,_),score in zip(records,scores)]


def old_v4_scores(records,matrices,final=False):
    """Frozen external comparator; never used as a new training feature."""
    priors=ROOT/'models/retrieval-priors'
    prefix='final' if final else 'evaluation'
    legacy=joblib.load(priors/f'{prefix}_legacy.joblib')
    semantic=joblib.load(priors/f'{prefix}_semantic.joblib')
    model_path=OLD_CACHE/('final_ranker_c0442eeba2d244f2.joblib' if final else 'lgb_rank_expanded_200_c0442eeba2d244f2.joblib')
    ranker=joblib.load(model_path)
    a=model_scores(legacy,[x[:,:32] for x in matrices])
    b=model_scores(semantic,[x[:,:38] for x in matrices])
    c=predict_scores(ranker,[x[:,:38] for x in matrices],200)
    result=[]
    for (_,base),la,se,ra in zip(records,a,b,c):
        lexical=score_candidates(base,best_config)
        first=.5/(60+rankdata(-la,method='min'))+.5/(60+rankdata(-lexical,method='min'))
        prior=.25/(60+rankdata(-se,method='min'))+.75/(60+rankdata(-first,method='min'))
        result.append(.75/(60+rankdata(-ra,method='min'))+.25/(60+rankdata(-prior,method='min')))
    return result


def blend_scores(scores,baseline,weight):
    if weight==0:
        return baseline
    if weight==1:
        return scores
    return [weight/(60+rankdata(-s,method='min'))+(1-weight)/(60+rankdata(-b,method='min'))
            for s,b in zip(scores,baseline)]


def matched_slice_recall(frame,recalls,target,mask=None):
    """Match query-length/filter/coordinate-presence strata within the actual slice."""
    from collections import Counter
    def strata(f):
        return list(zip(np.digitize(f.query_norm.str.len().to_numpy(),[15,25]).tolist(),
            (f.search_infm_params_text.str.len()>0).tolist(),f.search_location_id.isin(centers.index).tolist()))
    if mask is None:
        mask=np.ones(len(frame),dtype=bool)
    actual=strata(frame.loc[mask]); observed=Counter(actual); desired=Counter(strata(target))
    missing=set(desired)-set(observed)
    if missing:
        raise ValueError(f'Unsupported evaluation strata: {missing}')
    weights=np.array([desired.get(k,0)/observed[k] for k in actual],dtype=float)
    weights/=weights.sum()
    return float(recalls[mask]@weights)


def development_metrics(unseen_prediction,held_prediction,held_known):
    truths=labels_from_gold(gold,development)
    a=per_query_recall(unseen_prediction,truths)
    b=per_query_recall(held_prediction,truths)
    known=queries.query_norm.isin(history_all.query_norm)
    unknown_score=matched_slice_recall(development,a,queries[~known])
    known_score=matched_slice_recall(development,b,queries[known],held_known)
    return {'matched_recall50':(1-known_target)*unknown_score+known_target*known_score,
        'unseen_macro_recall50':float(a.mean()),'actual_known_macro_recall50':float(b[held_known].mean()),
        'held_context_macro_recall50':float(b.mean()),'held_actual_known_fraction':float(held_known.mean())}


def main():
    initial=prepare_v5_training(training,training_history,'evaluation')
    cold_model=fit_v5(initial,'cold_wide',mode='unseen_text')
    mixed_model=fit_v5(initial,'mixed_wide')
    del initial
    gc.collect()
    mined=prepare_v5_training(training,training_history,'evaluation',miner=mixed_model)
    mined_model=fit_v5(mined,'mixed_mined')
    del mined
    gc.collect()
    unseen_records,unseen_features,_,unseen_audit=evaluation_features(development,'unseen_text','development',control)
    held_records,held_features,held_known,held_audit=evaluation_features(development,'held_context','development',control)
    unseen_baseline=old_v4_scores(unseen_records,unseen_features)
    held_baseline=old_v4_scores(held_records,held_features)
    baseline=development_metrics(top50(unseen_records,unseen_baseline),top50(held_records,held_baseline),held_known)
    rows=[{'model':'v4','trees':200,'weight':0.,**baseline}]
    models={'cold_wide':cold_model,'mixed_wide':mixed_model,'mixed_mined':mined_model}
    for name,model in models.items():
        for trees in [100,200,400]:
            a=predict_scores(model,unseen_features,trees)
            b=predict_scores(model,held_features,trees)
            for weight in [.5,.75,1.]:
                metrics=development_metrics(top50(unseen_records,blend_scores(a,unseen_baseline,weight)),
                    top50(held_records,blend_scores(b,held_baseline,weight)),held_known)
                rows.append({'model':name,'trees':trees,'weight':weight,**metrics})
            print('Development',name,trees,'best',max(r['matched_recall50'] for r in rows if r['model']==name),flush=True)
    table=pd.DataFrame(rows).sort_values(['matched_recall50','model','trees','weight'],ascending=[False,True,True,True])
    table.to_csv(CACHE/'development_experiments.csv',index=False)
    admissible=table[table.unseen_macro_recall50>=baseline['unseen_macro_recall50']-.005]
    winner=admissible.iloc[0].to_dict()
    selection={'winner':winner,'baseline':baseline,'unseen_history':unseen_audit,'held_history':held_audit,
        'fingerprint':fingerprint,'config':V5_CONFIG,'feature_names':V5_FEATURE_NAMES,
        'control_used_for_selection':False,'frozen_comparator_limitation':'v4 auxiliary history may have used held control interactions'}
    (CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Selected on development',json.dumps(winner),flush=True)
    del unseen_records,unseen_features,held_records,held_features
    gc.collect()
    # Only now open the held control. It never changes the selected settings.
    records,features,_,audit=evaluation_features(control,'unseen_text','control',development)
    old_scores=old_v4_scores(records,features)
    if winner['model']=='v4':
        selected_scores=old_scores
    else:
        selected_scores=blend_scores(predict_scores(models[winner['model']],features,int(winner['trees'])),old_scores,float(winner['weight']))
    truth=labels_from_gold(gold,control)
    a=per_query_recall(top50(records,old_scores),truth)
    b=per_query_recall(top50(records,selected_scores),truth)
    random=np.random.default_rng(SEED+813)
    delta=b-a
    bootstrap=[float(random.choice(delta,len(delta),replace=True).mean()) for _ in range(4000)]
    result={'contexts':len(control),'v4_recall50':float(a.mean()),'selected_recall50':float(b.mean()),
        'delta':float(delta.mean()),'paired_delta_bootstrap95':np.quantile(bootstrap,[.025,.975]).tolist(),
        'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),
        'pool_recall':float(per_query_recall([r[0] for r in records],truth).mean()),'history_audit':audit,
        'used_for_selection':False,'frozen_comparator_limitation':selection['frozen_comparator_limitation']}
    (CACHE/'control_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('CONTROL',json.dumps(result),flush=True)
    final_fit_and_export(winner,models['mixed_wide'],selection,result)


def final_fit_and_export(winner,miner,selection,control_result):
    if winner['model']=='v4':
        print('No validated development improvement; kept original answer.csv.',flush=True)
        return
    frame=history_all[history_all.context_key.isin(gold)].drop_duplicates('context_key')
    frame=frame[[*QUERY_COLS,'query_norm','context_key']].sort_values('context_key').reset_index(drop=True)
    data=prepare_v5_training(frame,history_all,'final',miner=miner if winner['model']=='mixed_mined' else None)
    model=fit_v5(data,'final_ranker',mode='unseen_text' if winner['model']=='cold_wide' else None,trees=int(winner['trees']))
    del data
    gc.collect()
    history=V5History(history_all)
    path=CACHE/f'benchmark_{fingerprint}.joblib'
    if USE_CACHE and path.exists():
        records,features=joblib.load(path)
    else:
        records=retrieve_semantic_features(queries,history,progress_every=500)
        features=[v5_features(q,r,history) for q,r in zip(queries.itertuples(index=False),records)]
        save_cache((records,features),path)
    score=predict_scores(model,features,int(winner['trees']))
    if float(winner['weight'])<1:
        score=blend_scores(score,old_v4_scores(records,features,final=True),float(winner['weight']))
    prediction=top50(records,score)
    answer=pd.DataFrame({'query_id':queries.query_id.astype(str),
        'answer':[' '.join(ITEM_IDS[ids]) for ids in prediction]})
    assert list(answer.columns)==['query_id','answer'] and len(answer)==len(queries)
    assert answer.query_id.is_unique and set(answer.query_id)==set(queries.query_id)
    valid=set(ITEM_IDS)
    for text in answer.answer:
        values=text.split(' ')
        assert len(values)==50 and len(set(values))==50 and all(value in valid and re.fullmatch('[0-9a-f]{16}',value) for value in values)
    # Keep an explicit byte-identical copy of the previously submitted version.
    previous=ROOT/'answer_v0.4.csv'
    if not previous.exists():
        previous.write_bytes((ROOT/'answer.csv').read_bytes())
    answer.to_csv(CACHE/'answer_candidate.csv',index=False,encoding='utf-8',lineterminator='\n')
    loaded=pd.read_csv(CACHE/'answer_candidate.csv',dtype=str,keep_default_na=False)
    assert loaded.equals(answer)
    repeated=top50(records,blend_scores(predict_scores(model,features,int(winner['trees'])),
        old_v4_scores(records,features,final=True),float(winner['weight'])) if float(winner['weight'])<1 else predict_scores(model,features,int(winner['trees'])))
    assert all(np.array_equal(a,b) for a,b in zip(prediction,repeated))
    (ROOT/'answer.csv').write_bytes((CACHE/'answer_candidate.csv').read_bytes())
    manifest={'stage':'ranking-v5','fingerprint':fingerprint,'input_sha256':input_hashes,
        'pipeline_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selection':selection,'control':control_result,'final_contexts':len(frame),
        'feature_names':V5_FEATURE_NAMES,'answer_sha256':sha256_file(ROOT/'answer.csv'),
        'previous_answer_sha256':sha256_file(previous),'final_model':f'final_ranker_{fingerprint}.joblib',
        'versions':{'python':sys.version.split()[0],'lightgbm':lgb.__version__,'numpy':np.__version__,
                    'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('FINAL',json.dumps({'answer_sha256':manifest['answer_sha256'],'contexts':len(frame),'winner':winner}),flush=True)


if __name__=='__main__':
    main()
