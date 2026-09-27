# Final fitting uses the selected architecture and never tunes on audit results.
final_queries = pd.concat([expanded_training if best['scale']=='expanded' else training_queries,
    development, new_audit], ignore_index=True)[[*QUERY_COLS,'query_norm','context_key']]
final_queries = final_queries.drop_duplicates('context_key').sort_values('context_key').reset_index(drop=True)
final_model_files = []
final_model = None
if best['family'] != 'prior':
    final_path = CACHE / f"final_ranker_{fingerprint}.joblib"
    if USE_CACHE and final_path.exists():
        final_model = joblib.load(final_path)
    else:
        # Release evaluation matrices before the larger final training pass.
        del dataset, dev_features, seen_features, test_features
        gc.collect()
        final_data = grouped_training(history_all, final_queries, 'final_training')
        x,y,w,g = (final_data[key] for key in ['X','y','weight','group'])
        order = np.argsort(g,kind='stable')
        x,y,w,g = x[order],y[order],w[order],g[order]
        _,sizes = np.unique(g,return_counts=True)
        family = best['family']
        if family.startswith('lgb'):
            cls = lgb.LGBMRanker if family=='lgb_rank' else lgb.LGBMClassifier
            final_model = cls(n_estimators=best['trees'],num_leaves=31,learning_rate=.05,max_bin=127,
                min_child_samples=50,reg_lambda=10,random_state=SEED,n_jobs=8,
                verbosity=-1,deterministic=True,force_col_wise=True,
                **({'lambdarank_truncation_level':55,'label_gain':[0,1]} if family=='lgb_rank' else {}))
            final_model.fit(x,y,**({'group':sizes} if family=='lgb_rank' else {'sample_weight':w}))
        else:
            cls = CatBoostRanker if family=='cat_rank' else CatBoostClassifier
            final_model = cls(iterations=best['trees'],depth=6,learning_rate=.05,l2_leaf_reg=10,
                random_seed=SEED,thread_count=8,allow_writing_files=False,verbose=100,
                loss_function='YetiRank:mode=NDCG;top=50' if family=='cat_rank' else 'Logloss')
            final_model.fit(x,y,**({'group_id':g} if family=='cat_rank' else {'sample_weight':w}))
        save_cache(final_model,final_path)
        del final_data,x,y,w,g
        gc.collect()
    final_model_files.append(final_path.name)
prior_legacy = joblib.load(PRIORS_DIR/'final_legacy.joblib')
prior_semantic = joblib.load(PRIORS_DIR/'final_semantic.joblib')
benchmark_history = HistorySignals(history_all)
benchmark_records = cached_hybrid_records(queries,benchmark_history,'benchmark')
benchmark_features = build_rank_feature_records(queries,benchmark_records,'benchmark')
benchmark_reference = prior_scores(benchmark_records,benchmark_features)
if final_model is None:
    predictions = direct_predict(benchmark_records,benchmark_reference)
else:
    benchmark_scores = batched_new_scores(final_model,benchmark_features,best['family'])
    predictions = combined_predict(benchmark_records,benchmark_scores,benchmark_reference,best['weight'])
answer = pd.DataFrame({'query_id':queries.query_id.astype(str),
    'answer':[' '.join(ITEM_IDS[indices]) for indices in predictions]})
validate_answer(answer,queries.query_id,ITEM_IDS)
assert answer.answer.str.split().str.len().eq(50).all()
answer.to_csv(ROOT/'answer.csv',index=False,encoding='utf-8',lineterminator='\n')
reloaded = pd.read_csv(ROOT/'answer.csv',dtype=str,keep_default_na=False)
assert reloaded.equals(answer)
validate_answer(reloaded,queries.query_id,ITEM_IDS)
repeat = direct_predict(benchmark_records,benchmark_reference) if final_model is None else combined_predict(
    benchmark_records,batched_new_scores(final_model,benchmark_features,best['family']),benchmark_reference,best['weight'])
assert all(np.array_equal(a,b) for a,b in zip(predictions,repeat))
metrics = {'development':best,'reference_development':ref_metrics,'new_audit':result,
    'protocol':protocol,'known_query_fraction':known_fraction,'final_contexts':len(final_queries),
    'final_texts':int(final_queries.query_norm.nunique())}
(CACHE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
manifest = {'config':CONFIG,'experiment_config':EXPERIMENT_CONFIG,'ranker_choice':best,
    'semantic_config':SEMANTIC_CONFIG,'prior_manifest':prior_manifest,'frozen_priors':True,
    'model_manifest':model_manifest,'embedding_sha256':embedding_checksums,
    'input_sha256':input_hashes,'fingerprint':fingerprint,'code_sha256':code_hash,
    'final_model_files':final_model_files,'answer_sha256':sha256_file(ROOT/'answer.csv'),'metrics':metrics,
    'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,
        'sklearn':sklearn.__version__,'torch':torch.__version__,'transformers':transformers.__version__,
        'lightgbm':lgb.__version__,'catboost':__import__('catboost').__version__,'device':DEVICE}}
(CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Saved',ROOT/'answer.csv','; queries',len(answer),'; SHA256',manifest['answer_sha256'])
