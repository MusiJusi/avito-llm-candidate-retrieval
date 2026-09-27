"""Compare stronger rankers on the exact v7 OOF features and frozen gold.

Each configuration is content-addressed and resumable. Candidate pools, labels,
query encoders and history exclusions are unchanged, so this experiment isolates
the ranker. Development selects one variant; the previously viewed control is
reported afterwards. No hidden labels or search_category scoring are used.
"""
from pathlib import Path
import ast
import hashlib
import json
import gc
import time

DRIVER = Path(__file__).resolve()
source = DRIVER.with_name('query_encoder_rank.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if not (isinstance(node, ast.If)
    and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
    and node.test.left.id == '__name__')]
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())
__file__ = str(DRIVER)

V8_CACHE = ROOT/'artifacts/ranker-v8'
V8_CACHE.mkdir(exist_ok=True)
V8_FP = hashlib.sha256((QR_FP+hashlib.sha256(DRIVER.read_bytes()).hexdigest()).encode()).hexdigest()[:16]
CONFIGURATIONS = {
    'longer31': dict(n_estimators=800, num_leaves=31, learning_rate=.03, reg_lambda=10., cutoff=55),
    'leaves63': dict(n_estimators=600, num_leaves=63, learning_rate=.04, reg_lambda=20., cutoff=55),
    'cutoff100': dict(n_estimators=600, num_leaves=31, learning_rate=.04, reg_lambda=10., cutoff=100),
    'regularized': dict(n_estimators=600, num_leaves=31, learning_rate=.04, reg_lambda=30., cutoff=55),
    'xendcg': dict(n_estimators=600, num_leaves=31, learning_rate=.04, reg_lambda=10., objective='rank_xendcg'),
}

def fitted_model(data, extra, name, configuration, stage='evaluation'):
    path = V8_CACHE/f'{stage}_{name}_{V8_FP}.joblib'
    if path.exists():
        return joblib.load(path)
    parameters = dict(configuration)
    cutoff = parameters.pop('cutoff', None)
    model = lgb.LGBMRanker(max_bin=127, min_child_samples=50, random_state=SEED,
        n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True,
        label_gain=[0, 1], **parameters)
    if cutoff is not None:
        model.set_params(lambdarank_truncation_level=cutoff)
    x = np.column_stack([data['X'], extra])
    assert x.shape[1] == 51 and len(x) == len(data['y'])
    print('Fit v8', stage, name, x.shape, flush=True)
    started = time.perf_counter()
    model.fit(x, data['y'], group=data['sizes'],
        sample_weight=np.repeat(data['group_weight'], data['sizes']))
    save_cache(model, path)
    del x
    gc.collect()
    report = {'stage': stage, 'configuration': configuration, 'rows': len(data['y']),
        'groups': len(data['sizes']), 'seconds': round(time.perf_counter()-started, 2),
        'model_sha256': sha256_file(path)}
    (path.with_suffix('.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Fit completed', name, json.dumps(report), flush=True)
    return model

def prediction_variants(frame, mode, stage, models, other_frame):
    """One pool at a time to keep the 32 GB laptop memory budget practical."""
    records, features, known, audit = evaluation_features(frame, mode, stage, other_frame)
    vectors = evaluation_query_features(frame, 'development' if stage=='development' else 'control')
    lookup = {text:i for i,text in enumerate(sorted(set(frame.query_norm)))}
    matrices = [np.column_stack([x, query_features(vectors[lookup[text]], ids, raw)])
        for text, (ids, raw), x in zip(frame.query_norm, records, features)]
    original = baseline_v5(records, features)
    truth = labels_from_gold(gold, frame)
    predictions = {}
    for name, model in models.items():
        for trees in sorted(set([model.n_estimators//2, model.n_estimators])):
            score = predict_scores(model, matrices, trees)
            for weight in [.5, .75, 1.]:
                prediction = top50(records, blend_scores(score, original, weight))
                predictions[(name, trees, weight)] = per_query_recall(prediction, truth)
    reference = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    prediction = top50(records, blend_scores(predict_scores(reference, matrices, 400), original, .75))
    predictions[('v7', 400, .75)] = per_query_recall(prediction, truth)
    del records, features, matrices, original
    gc.collect()
    return predictions, known, audit

def run_comparison():
    data, extra = prepare_query_oof()
    models = {name:fitted_model(data, extra, name, config) for name,config in CONFIGURATIONS.items()}
    del data, extra
    gc.collect()
    unseen, _, unseen_audit = prediction_variants(development, 'unseen_text', 'development', models, control)
    held, known, held_audit = prediction_variants(development, 'held_context', 'development', models, control)
    query_known = queries.query_norm.isin(history_all.query_norm)
    rows = []
    for key in unseen:
        a = matched_slice_recall(development, unseen[key], queries[~query_known])
        b = matched_slice_recall(development, held[key], queries[query_known], known)
        rows.append({'model':key[0], 'trees':key[1], 'weight':key[2],
            'matched_recall50':(1-known_target)*a+known_target*b,
            'unseen_macro_recall50':float(unseen[key].mean()),
            'held_context_macro_recall50':float(held[key].mean()),
            'actual_known_macro_recall50':float(held[key][known].mean())})
    table = pd.DataFrame(rows).sort_values(['matched_recall50','model','trees','weight'],
        ascending=[False,True,True,True])
    table.to_csv(V8_CACHE/'development.csv', index=False)
    reference = table[table.model=='v7'].iloc[0].to_dict()
    assert abs(reference['matched_recall50']-.9501708535101647)<1e-10
    # Freeze the winner before reading control results. Avoid trading away new texts.
    eligible = table[table.unseen_macro_recall50>=reference['unseen_macro_recall50']-.002]
    winner = eligible.iloc[0].to_dict()
    key = (winner['model'], int(winner['trees']), float(winner['weight']))
    baseline_key = ('v7', 400, .75)
    random = np.random.default_rng(SEED+1001)
    paired = (1-known_target)*(unseen[key]-unseen[baseline_key])+known_target*(held[key]-held[baseline_key])
    # This unweighted paired interval supplements, rather than replaces, matched selection.
    bootstrap = [float(random.choice(paired,len(paired),replace=True).mean()) for _ in range(3000)]
    selection = {'fingerprint':V8_FP, 'winner':winner, 'baseline':reference,
        'configuration':CONFIGURATIONS.get(winner['model']), 'control_used_for_selection':False,
        'development_pair_delta_bootstrap95':np.quantile(bootstrap,[.025,.975]).tolist(),
        'unseen_history':unseen_audit, 'held_history':held_audit,
        'limitations':['Repeatedly viewed development/control.', 'Frozen v4 auxiliary priors limitation.',
                      'Evaluation query encoder excludes all development texts.']}
    (V8_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('V8 selected', json.dumps(selection), flush=True)
    records, features, _, audit = evaluation_features(control,'unseen_text','control',development)
    vectors = evaluation_query_features(control,'control')
    lookup = {text:i for i,text in enumerate(sorted(set(control.query_norm)))}
    matrices = [np.column_stack([x,query_features(vectors[lookup[text]],ids,raw)])
        for text,(ids,raw),x in zip(control.query_norm,records,features)]
    original = baseline_v5(records,features)
    before_model = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    chosen = before_model if winner['model']=='v7' else models[winner['model']]
    truths = labels_from_gold(gold,control)
    before = per_query_recall(top50(records,blend_scores(predict_scores(before_model,matrices,400),original,.75)),truths)
    after = per_query_recall(top50(records,blend_scores(predict_scores(chosen,matrices,key[1]),original,key[2])),truths)
    report = {'v7_recall50':float(before.mean()),'selected_recall50':float(after.mean()),
        'delta':float((after-before).mean()),'improved':int((after>before).sum()),
        'worsened':int((after<before).sum()),'history_audit':audit,
        'used_for_configuration_selection':False,'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH}
    (V8_CACHE/'control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert report['main_answer_unchanged']
    print('V8 control',json.dumps(report),flush=True)

if __name__=='__main__':
    run_comparison()
