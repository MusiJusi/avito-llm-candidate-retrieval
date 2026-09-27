"""Check the development-selected E5 pilot on control without tuning settings.

This check uses the unchanged v5 candidate pools and ranker. It measures the
additional query score, not recall of a newly generated retrieval pool.
"""
from pathlib import Path
import ast
import json
import gc

driver = Path(__file__).resolve().with_name('query_encoder_pilot.py')
tree = ast.parse(driver.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n, ast.If)
    and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
    and n.test.left.id == '__name__')]
__file__ = str(driver)
exec(compile(tree, str(driver), 'exec'), globals())


def check_query_control():
    selection = json.loads((QE_CACHE/'pilot_report.json').read_text(encoding='utf-8'))
    winner = selection['winner']
    assert winner['variant'].startswith('epoch_')
    epoch = int(winner['variant'].split('_')[1])
    model = AutoModel.from_pretrained(QE_CACHE/f'epoch_{epoch}_{QE_FP}',
        local_files_only=True, attn_implementation='eager').to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    texts = sorted(set(control.query_norm))
    vectors = encode_learned(model, tokenizer, texts)
    lookup = {text: i for i, text in enumerate(texts)}
    save_array(vectors, QE_CACHE/f'control_queries_epoch_{epoch}_{QE_FP}.npy')
    del model, tokenizer
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    records, features, known, audit = evaluation_features(control, 'unseen_text', 'control', development)
    original = baseline_v5(records, features)
    scores = []
    for query, (ids, base) in zip(control.query_norm, records):
        cosine = (semantic_index.items[ids].astype(np.float64)
            @ vectors[lookup[query]].astype(np.float64)).astype(np.float32)
        scores.append(semantic_affinity(cosine, base[:,4]) if winner['affinity']=='geo' else cosine)
    chosen = blend_scores(scores, original, float(winner['weight']))
    truth = labels_from_gold(gold, control)
    before = per_query_recall(top50(records, original), truth)
    after = per_query_recall(top50(records, chosen), truth)
    delta = after-before
    random = np.random.default_rng(SEED+907)
    bootstrap = [float(random.choice(delta, len(delta), replace=True).mean()) for _ in range(4000)]
    result = {'fingerprint': QE_FP, 'selection': winner, 'contexts': len(control),
        'v5_recall50': float(before.mean()), 'query_encoder_recall50': float(after.mean()),
        'delta': float(delta.mean()), 'improved': int((delta>0).sum()),
        'worsened': int((delta<0).sum()), 'bootstrap95': np.quantile(bootstrap,[.025,.975]).tolist(),
        'used_for_selection': False, 'history_audit': audit,
        'answer_unchanged': sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation': 'Previously viewed control; frozen v4 auxiliary-history limitation. Fixed v5 pools; no OOF downstream refit.'}
    assert result['answer_unchanged']
    (QE_CACHE/'control_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('QUERY CONTROL',json.dumps(result),flush=True)


if __name__=='__main__':
    check_query_control()
