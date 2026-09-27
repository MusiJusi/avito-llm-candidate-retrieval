"""Test a pretrained CE as a learned feature, reusing the earlier pilot scores.

This is an economical, explicitly limited pilot. A text-grouped split of the
previously viewed development pilot trains a small second selector. Its upstream
v7 models excluded all development texts. A matched selector without CE isolates
the benefit of the CE features. No control or benchmark labels are inspected.
Positive results require genuine training OOF shortlists before final refit.
"""
from pathlib import Path
import ast
import json
import hashlib
import gc

DRIVER = Path(__file__).resolve()
source = DRIVER.with_name('query_encoder_rank.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__ = str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__ = str(DRIVER)
CE_FEATURE_CACHE = ROOT/'artifacts/ce-feature-v8'
CE_FEATURE_CACHE.mkdir(exist_ok=True)

def run_feature_pilot():
    model = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    vectors = evaluation_query_features(development,'development')
    lookup = {text:i for i,text in enumerate(sorted(set(development.query_norm)))}
    pools = []
    for mode in ['unseen_text','held_context']:
        files = list((ROOT/'artifacts/cross-encoder-pilot').glob(f'{mode}_*.joblib'))
        assert len(files)==1,'Pin one existing frozen CE pilot rather than mixing runs'
        bundle = joblib.load(files[0])
        _,features,_,audit = evaluation_features(development,mode,'development',control)
        for i,(position,record,selected,logits) in enumerate(zip(bundle['positions'],bundle['records'],bundle['selected'],bundle['logits'])):
            query = development.iloc[position]
            ids,raw = record
            x = np.column_stack([features[position],query_features(vectors[lookup[query.query_norm]],ids,raw)])
            score = blend_scores([predict_scores(model,[x],400)[0]],[bundle['baseline'][i]],.75)[0]
            shortlist = stable_topk(score,300)
            ce = np.full(len(ids),np.nan,np.float32)
            ce[selected] = logits
            ce_rank = np.full(len(ids),np.nan,np.float32)
            ce_rank[selected] = np.log1p(rankdata(-logits,method='min'))
            stage_rank = np.log1p(rankdata(-score,method='min')).astype(np.float32)
            base = np.column_stack([x[shortlist],stage_rank[shortlist]])
            expanded = np.column_stack([base,ce[shortlist],ce_rank[shortlist]])
            truth = gold[query.context_key]
            target = np.isin(ids[shortlist],list(truth)).astype(np.int32)
            pools.append({'mode':mode,'query_norm':query.query_norm,'base':base,'expanded':expanded,
                'y':target,'ids':ids[shortlist],'truth':truth,'v7':score[shortlist],
                'ce_coverage':float(np.isfinite(ce[shortlist]).mean())})
        del bundle,features
        gc.collect()
    unique = sorted({p['query_norm'] for p in pools})
    random = np.random.default_rng(SEED+1003)
    held_texts = set(random.choice(unique,max(1,int(.3*len(unique))),replace=False))
    train_pools = [p for p in pools if p['query_norm'] not in held_texts]
    validation_pools = [p for p in pools if p['query_norm'] in held_texts]
    assert not {p['query_norm'] for p in train_pools}&{p['query_norm'] for p in validation_pools}
    variants = {}
    for variant in ['base','expanded']:
        x = np.concatenate([p[variant] for p in train_pools]).astype(np.float32)
        y = np.concatenate([p['y'] for p in train_pools])
        groups = np.array([len(p['y']) for p in train_pools])
        weights = np.repeat([1/max(int(p['y'].sum()),1) for p in train_pools],groups)
        selector = lgb.LGBMRanker(n_estimators=200,num_leaves=7,learning_rate=.03,reg_lambda=30,
            min_child_samples=100,max_bin=127,lambdarank_truncation_level=55,label_gain=[0,1],
            random_state=SEED,n_jobs=8,verbosity=-1,deterministic=True,force_col_wise=True)
        selector.fit(x,y,group=groups,sample_weight=weights)
        variants[variant] = predict_scores(selector,[p[variant] for p in validation_pools],200)
        del x,y,selector
        gc.collect()
    truth = [p['truth'] for p in validation_pools]
    records = [(p['ids'],None) for p in validation_pools]
    before = per_query_recall(top50(records,[p['v7'] for p in validation_pools]),truth)
    rows = []
    paired = {}
    known_share = known_target
    mode = np.array([p['mode'] for p in validation_pools])
    for variant in ['base','expanded']:
        for weight in [.25,.5,1.]:
            score = blend_scores(variants[variant],[p['v7'] for p in validation_pools],weight)
            after = per_query_recall(top50(records,score),truth)
            a = float(after[mode=='unseen_text'].mean());b=float(after[mode=='held_context'].mean())
            rows.append({'variant':variant,'weight':weight,'mixed_recall50':(1-known_share)*a+known_share*b,
                'unseen_recall50':a,'held_recall50':b,'improved':int((after>before).sum()),
                'worsened':int((after<before).sum())})
            paired[(variant,weight)] = after-before
    a = float(before[mode=='unseen_text'].mean());b=float(before[mode=='held_context'].mean())
    baseline = {'variant':'v7','weight':0.,'mixed_recall50':(1-known_share)*a+known_share*b,
        'unseen_recall50':a,'held_recall50':b,'improved':0,'worsened':0}
    rows.append(baseline)
    table = pd.DataFrame(rows).sort_values('mixed_recall50',ascending=False)
    table.to_csv(CE_FEATURE_CACHE/'pilot_ablation.csv',index=False)
    best_without = table[table.variant!='expanded'].iloc[0].to_dict()
    best_with = table[table.variant=='expanded'].iloc[0].to_dict()
    report = {'training_contexts':len(train_pools),'validation_contexts':len(validation_pools),
        'validation_texts':len(held_texts),'own_text_overlap':0,'baseline':baseline,
        'best_without_ce':best_without,'best_with_ce':best_with,
        'ce_delta_vs_best_without':best_with['mixed_recall50']-best_without['mixed_recall50'],
        'mean_ce_score_coverage':float(np.mean([p['ce_coverage'] for p in validation_pools])),
        'control_evaluated':False,'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation':'Small grouped split of previously viewed development; score coverage follows v5 top500; pilot only.'}
    assert report['main_answer_unchanged']
    (CE_FEATURE_CACHE/'pilot_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CE feature pilot',json.dumps(report),flush=True)

if __name__=='__main__':
    run_feature_pilot()
