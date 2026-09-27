"""Evaluate whether learned full-corpus retrieval improves actual Recall@50.

The frozen v7 selector scores the expanded pool. This is a retrieval ablation;
candidate features are recomputed over the new pool consistently. Training a
selector specifically on expanded OOF pools remains a separate next experiment.
"""
from pathlib import Path
import ast
import json
import gc

DRIVER = Path(__file__).resolve()
source = DRIVER.with_name('query_encoder_rank.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__ = str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__ = str(DRIVER)
EXPANDED_CACHE = ROOT/'artifacts/expanded-pool-v8'
EXPANDED_CACHE.mkdir(exist_ok=True)
LEARNED_LOOKUP = {}

def install_expanded_retriever():
    document = json.loads((ROOT/'Avito.ipynb').read_text(encoding='utf-8'))
    original = None
    for cell in document['cells']:
        if cell['cell_type']!='code':continue
        text = ''.join(cell['source'])
        for node in ast.parse(text).body:
            if isinstance(node,ast.FunctionDef) and node.name=='retrieve_semantic_features':
                original = ast.get_source_segment(text,node)
    assert original is not None
    # Preserve the original first-stage score columns and floating-point arithmetic.
    changed = original.replace('def retrieve_semantic_features(', 'def retrieve_expanded_features(')
    changed = changed.replace('cosine = dense_block[i % 64]', '''cosine = dense_block[i % 64]
        if i % 64 == 0:
            learned_vectors = np.stack([LEARNED_LOOKUP[text] for text in texts[i:i+64]])
            learned_block = (learned_vectors.astype(np.float64) @ semantic_index.item_transpose).astype(np.float32)
        learned_cosine = learned_block[i % 64]''')
    changed = changed.replace('candidates = np.union1d(lexical_candidates, semantic_candidates)', '''learned_candidates = np.union1d(stable_topk(learned_cosine, 500),
                                          stable_topk(semantic_affinity(learned_cosine, geo), 500))
        semantic_candidates = np.union1d(semantic_candidates, learned_candidates)
        candidates = np.union1d(lexical_candidates, semantic_candidates)''')
    assert 'learned_candidates' in changed and 'learned_block' in changed
    exec(compile(changed,str(DRIVER),'exec'),globals())
    return changed

def evaluate_expansion():
    global LEARNED_LOOKUP
    install_expanded_retriever()
    vectors = evaluation_query_features(development,'development')
    LEARNED_LOOKUP = dict(zip(sorted(set(development.query_norm)),vectors))
    ranker = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    truth = labels_from_gold(gold,development)
    old = pd.read_csv(QR_CACHE/'development_query_deltas.csv')
    before = {}
    after = {}
    saved_known = None
    statistics = []
    for mode in ['unseen_text','held_context']:
        history_frame = evaluation_history(development,mode,control)
        known = development.query_norm.isin(history_frame.query_norm).to_numpy()
        history = V5History(history_frame)
        del history_frame
        records = retrieve_expanded_features(development,history,progress_every=640)
        features = [v5_features(query,record,history) for query,record in zip(development.itertuples(index=False),records)]
        del history
        baseline = baseline_v5(records,features)
        matrices = [np.column_stack([x,query_features(LEARNED_LOOKUP[text],ids,raw)])
            for text,(ids,raw),x in zip(development.query_norm,records,features)]
        scores = blend_scores(predict_scores(ranker,matrices,400),baseline,.75)
        after[mode] = per_query_recall(top50(records,scores),truth)
        old_rows = old[old['mode']==mode].set_index('context_key')
        before[mode] = old_rows.loc[development.context_key,'query_recall50'].to_numpy()
        if mode=='held_context':saved_known=known
        statistics.append({'mode':mode,'before_macro_recall50':float(before[mode].mean()),
            'after_macro_recall50':float(after[mode].mean()),
            'improved':int((after[mode]>before[mode]).sum()),'worsened':int((after[mode]<before[mode]).sum()),
            'pool_recall':float(per_query_recall([r[0] for r in records],truth).mean()),
            'mean_pool_size':float(np.mean([len(r[0]) for r in records]))})
        pd.DataFrame({'context_key':development.context_key,'query':development.query_norm,
            'before_recall50':before[mode],'after_recall50':after[mode]}).to_csv(EXPANDED_CACHE/f'{mode}_queries.csv',index=False)
        del records,features,matrices,baseline,scores
        gc.collect()
    query_known = queries.query_norm.isin(history_all.query_norm)
    def matched(values):
        return (1-known_target)*matched_slice_recall(development,values['unseen_text'],queries[~query_known])+known_target*matched_slice_recall(development,values['held_context'],queries[query_known],saved_known)
    report = {'before_matched_recall50':matched(before),'after_matched_recall50':matched(after),
        'statistics':statistics,'control_evaluated':False,
        'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation':'Fixed v7 selector applied to expanded pools; OOF pool refit not performed.'}
    assert abs(report['before_matched_recall50']-.9501708535101647)<1e-10
    assert report['main_answer_unchanged']
    (EXPANDED_CACHE/'pilot_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Expanded pool rank pilot',json.dumps(report),flush=True)

if __name__=='__main__':
    evaluate_expansion()
