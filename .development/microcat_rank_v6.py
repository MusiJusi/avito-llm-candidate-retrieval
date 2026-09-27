"""OOF microcat features, isolated ranker ablations, and held-control reporting.

Reconstructs the exact v5 mined groups from cached pools. Equality assertions
ensure the only difference in each ranker ablation is the added feature family.
The auxiliary classifier never sees the context whose ranker features it makes.
"""
from pathlib import Path
import ast
import json
import hashlib
import gc
import sys
import time

RANK_DRIVER = Path(__file__).resolve()
source = RANK_DRIVER.with_name('microcat_v6.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                                       and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')]
__file__ = str(source)
exec(compile(tree, str(source), 'exec'), globals())
__file__ = str(RANK_DRIVER)
RANK_FP = hashlib.sha256((MICRO_FP + hashlib.sha256(RANK_DRIVER.read_bytes()).hexdigest()).encode()).hexdigest()[:16]
ITEM_MICROCATS = items.item_microcat_id.to_numpy(dtype=np.int64)
VARIANTS = {'nb': list(range(7)), 'mlp': list(range(7, 14)), 'both': list(range(14))}


def training_micro_features(space, frame, source_history, stage):
    output = MICRO_CACHE/f'{stage}_oof_features_{RANK_FP}.joblib'
    data = joblib.load(V5_CACHE/f'{stage}_mined_{V5_FP}.joblib')
    if USE_CACHE and output.exists():
        extra = joblib.load(output)
        assert extra.shape == (len(data['y']), 14)
        return data, extra
    # No new retrieval/sampling policy: frozen pilot and random draws match v5.
    miner = joblib.load(V5_CACHE/f'mixed_wide_{V5_FP}.joblib')
    random = np.random.default_rng(SEED+811)
    truths = labels_from_gold(gold, frame)
    extra = np.empty((len(data['y']), 14), np.float32)
    group, cursor = 0, 0
    audits = []
    for mode in ['unseen_text', 'held_context']:
        assignments = oof_assignments(frame, mode, 3, SEED+803)
        for fold in range(3):
            positions = np.flatnonzero(assignments == fold)
            fold_queries = frame.iloc[positions]
            fold_truths = [truths[i] for i in positions]
            fit, cold, _ = history_for_queries(source_history, fold_queries, fold_truths,
                ITEM_IDS, mode, .9, SEED+804+fold)
            assert not set(fold_queries.context_key) & set(fit.context_key)
            models, predictions = {}, {}
            for kind in ['nb', 'mlp']:
                models[kind] = classifier(space, fit, f'{stage}_{mode}_{fold}', kind)
                predictions[kind] = predict_micro(models[kind], space, fold_queries)
            audits.append({'mode': mode, 'fold': fold, 'classifier_pairs': len(fit),
                           'classifier_history_sha256': history_digest(fit), 'own_context_overlap': 0})
            del fit
            for start in range(0, len(positions), V5_CONFIG['chunk']):
                path = V5_CACHE/f'{stage}_pool_{mode}_{fold}_{start}_{V5_FP}.joblib'
                assert path.exists(), f'Rebuild v5 sampled pools first: {path.name}'
                records, matrices = joblib.load(path)
                for offset, ((ids, base), features) in enumerate(zip(records, matrices)):
                    selected, target = sample_v5_group(ids, base, features, fold_truths[start+offset], random, miner)
                    if not len(selected):
                        continue
                    size = len(selected)
                    assert data['query_position'][group] == positions[start+offset]
                    assert data['mode'][group] == mode and data['sizes'][group] == size
                    assert np.array_equal(data['y'][cursor:cursor+size], target[selected])
                    assert np.array_equal(data['X'][cursor:cursor+size], features[selected], equal_nan=True)
                    columns = []
                    for kind in ['nb', 'mlp']:
                        columns.append(micro_candidate_features(predictions[kind][start+offset],
                            ITEM_MICROCATS[ids[selected]], space['classes'], models[kind]['supported']))
                    extra[cursor:cursor+size] = np.column_stack(columns)
                    cursor += size
                    group += 1
                if start % 1024 == 0:
                    print('Micro OOF',stage,mode,fold,'processed',min(start+128,len(positions)),'rows',cursor,flush=True)
                del records, matrices
            del models, predictions
            gc.collect()
    assert cursor == len(data['y']) and group == len(data['sizes'])
    save_cache(extra, output)
    (MICRO_CACHE/f'{stage}_oof_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    return data, extra


def fit_micro_ranker(data, extra, variant, stage, trees=400):
    path = MICRO_CACHE/f'{stage}_rank_{variant}_{RANK_FP}.joblib'
    if USE_CACHE and path.exists():
        return joblib.load(path)
    columns = VARIANTS[variant]
    x = np.column_stack([data['X'], extra[:, columns]])
    model = lgb.LGBMRanker(n_estimators=trees, num_leaves=31, learning_rate=.05,
        max_bin=127, min_child_samples=50, reg_lambda=10, random_state=SEED, n_jobs=8,
        verbosity=-1, deterministic=True, force_col_wise=True,
        lambdarank_truncation_level=55, label_gain=[0,1])
    started = time.perf_counter()
    print('Micro ranker',stage,variant,'rows',len(x),'features',x.shape[1],flush=True)
    model.fit(x, data['y'], group=data['sizes'], sample_weight=np.repeat(data['group_weight'],data['sizes']))
    save_cache(model, path)
    print('Micro ranker fitted',variant,'seconds',round(time.perf_counter()-started,1),flush=True)
    del x
    gc.collect()
    return model


def augmented_evaluation(frame, mode, stage):
    other = control if stage == 'development' else development
    records, matrices, known, audit = evaluation_features(frame, mode, stage, other)
    probabilities = joblib.load(MICRO_CACHE/f'{stage}_{mode}_probabilities_{MICRO_FP}.joblib')
    features = []
    for position, ((ids, _), base) in enumerate(zip(records, matrices)):
        added = [micro_candidate_features(probabilities[kind]['probabilities'][position],
            ITEM_MICROCATS[ids], np.sort(history_all.item_microcat_id.unique()), probabilities[kind]['supported'])
            for kind in ['nb','mlp']]
        features.append(np.column_stack([base, *added]))
    return records, matrices, features, known, audit


def selected_columns(variant):
    return list(range(45)) + [45+i for i in VARIANTS[variant]]


def run_experiment():
    space = prepare_micro_space()
    # These predictions are fitted independently of ranker training; control stays closed.
    for mode in ['unseen_text','held_context']:
        path = MICRO_CACHE/f'development_{mode}_probabilities_{MICRO_FP}.joblib'
        if not path.exists():
            evaluation_micro(space, development, mode, 'development', control)
    data, extra = training_micro_features(space, training, training_history, 'evaluation')
    models = {variant: fit_micro_ranker(data,extra,variant,'evaluation') for variant in VARIANTS}
    del data, extra
    gc.collect()
    bundles = {}
    for mode in ['unseen_text','held_context']:
        bundle = augmented_evaluation(development,mode,'development')
        records, base, features, known, audit = bundle
        score_path = MICRO_CACHE/f'development_{mode}_v5scores_{MICRO_FP}.joblib'
        if USE_CACHE and score_path.exists():
            base_scores = joblib.load(score_path)
        else:
            base_scores = baseline_v5(records, base)
            save_cache(base_scores, score_path)
        bundles[mode] = (*bundle, base_scores)
    a = bundles['unseen_text']; b = bundles['held_context']
    baseline = development_metrics(top50(a[0],a[5]),top50(b[0],b[5]),b[3])
    rows = [{'variant':'v5','trees':400,'weight':0.,**baseline}]
    for variant, model in models.items():
        cols = selected_columns(variant)
        for trees in [200,400]:
            first = predict_scores(model,[x[:,cols] for x in a[2]],trees)
            second = predict_scores(model,[x[:,cols] for x in b[2]],trees)
            for weight in [.5,.75,1.]:
                metrics = development_metrics(top50(a[0],blend_scores(first,a[5],weight)),
                    top50(b[0],blend_scores(second,b[5],weight)),b[3])
                rows.append({'variant':variant,'trees':trees,'weight':weight,**metrics})
            print('Micro development',variant,trees,'best',max(x['matched_recall50'] for x in rows if x['variant']==variant),flush=True)
    table = pd.DataFrame(rows).sort_values(['matched_recall50','variant','trees','weight'], ascending=[False,True,True,True])
    table.to_csv(MICRO_CACHE/'ranker_ablation.csv',index=False,lineterminator='\n')
    allowed = table[table.unseen_macro_recall50 >= baseline['unseen_macro_recall50']-.002]
    winner = allowed.iloc[0].to_dict()
    selection = {'winner':winner,'baseline':baseline,'micro_fingerprint':MICRO_FP,'rank_fingerprint':RANK_FP,
        'control_used_for_selection':False,'base_answer_sha256':BASE_ANSWER_HASH,
        'v5_comparator_limitation':V5_MANIFEST['selection']['frozen_comparator_limitation'],
        'feature_families':{k:[f'{k}_{name}' for name in MICRO_FEATURE_NAMES] for k in ['nb','mlp']},
        'ablation': 'Same mined candidate groups, positives, weights and ranker hyperparameters; only microcat features differ.'}
    (MICRO_CACHE/'ranker_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('MICRO SELECTION',json.dumps(selection),flush=True)
    del bundles,a,b
    gc.collect()
    # Open held control only after every setting has been selected and saved.
    evaluation_micro(space,control,'unseen_text','control',development)
    records,base,features,_,audit = augmented_evaluation(control,'unseen_text','control')
    v5_scores = baseline_v5(records,base)
    if winner['variant']=='v5':
        chosen = v5_scores
    else:
        chosen = blend_scores(predict_scores(models[winner['variant']],
            [x[:,selected_columns(winner['variant'])] for x in features],int(winner['trees'])),v5_scores,float(winner['weight']))
    truth = labels_from_gold(gold,control)
    old = per_query_recall(top50(records,v5_scores),truth)
    new = per_query_recall(top50(records,chosen),truth)
    delta = new-old
    random = np.random.default_rng(SEED+902)
    boot = [float(random.choice(delta,len(delta),replace=True).mean()) for _ in range(4000)]
    result = {'v5_recall50':float(old.mean()),'micro_recall50':float(new.mean()),'delta':float(delta.mean()),
        'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),
        'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'contexts':len(control),
        'used_for_selection':False,'history_audit':audit,
        'limitation': 'Control has already been evaluated in v5; frozen v4 priors also have an auxiliary-history limitation.'}
    (MICRO_CACHE/'ranker_control.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert sha256_file(ROOT/'answer.csv') == BASE_ANSWER_HASH
    print('MICRO CONTROL',json.dumps(result),flush=True)


if __name__ == '__main__':
    run_experiment()
