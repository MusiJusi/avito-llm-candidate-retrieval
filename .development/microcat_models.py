"""Local query/filter classifiers; no category, geography or item IDs as inputs.

Duplicate interactions are aggregated into soft labels. Both models retain every
observed microcat for a text/filter input rather than choosing a single label.
The feature-only E5 encoder is frozen. Model fitting receives an isolated history.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch


MICRO_FEATURE_NAMES = ['micro_probability', 'micro_log_probability', 'micro_log_rank',
                       'micro_peak', 'micro_entropy', 'micro_margin', 'micro_supported']


def micro_inputs(frame):
    """Location and search_category intentionally do not participate in keys."""
    query = frame.query_norm.astype(str)
    filters = frame.search_infm_params_text.fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    return pd.DataFrame({'query': query, 'filters': filters,
                         'key': [json.dumps([q, f], ensure_ascii=False) for q, f in zip(query, filters)]}, index=frame.index)


def input_digest(frame):
    return hashlib.sha256('\n'.join(micro_inputs(frame).key).encode()).hexdigest()[:16]


def aggregated_targets(history, classes, input_to_row):
    """Use all supplied interactions, including positives outside benchmark_items."""
    inputs = micro_inputs(history)
    pairs = pd.DataFrame({'row': inputs.key.map(input_to_row).to_numpy(),
                          'label': history.item_microcat_id.to_numpy()})
    counts = pairs.groupby(['row', 'label'], sort=True).size().reset_index(name='count')
    rows = np.sort(counts.row.unique()).astype(np.int32)
    row_to_local = {int(row): i for i, row in enumerate(rows)}
    class_to_column = {int(label): i for i, label in enumerate(classes)}
    targets = np.zeros((len(rows), len(classes)), np.float32)
    for row, label, count in counts.itertuples(index=False, name=None):
        targets[row_to_local[int(row)], class_to_column[int(label)]] = count
    totals = targets.sum(axis=1)
    assert (totals > 0).all()
    supported = targets.sum(axis=0) > 0
    return rows, targets, totals, supported


def sparse_inputs(inputs):
    """Stateless, positive hashed n-grams: no vocabulary fitted on held labels."""
    char = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5),
        n_features=2**16, alternate_sign=False, norm='l1', dtype=np.float32)
    word = HashingVectorizer(ngram_range=(1, 2), n_features=2**15,
        alternate_sign=False, norm='l1', dtype=np.float32)
    return sparse.hstack([char.transform(inputs['query']), word.transform(inputs['query']),
                         .5 * word.transform(inputs.filters)], format='csr', dtype=np.float32)


def fit_nb(space, history):
    rows, targets, totals, supported = aggregated_targets(history, space['classes'], space['input_to_row'])
    local_row, labels = np.nonzero(targets)
    # Equal input weight with a modest amount-of-evidence weight, never 1-hot collapse.
    weights = targets[local_row, labels] / totals[local_row] * np.minimum(np.sqrt(totals[local_row]), 16)
    model = MultinomialNB(alpha=.005)
    model.partial_fit(space['sparse'][rows[local_row]], space['classes'][labels],
                      classes=space['classes'], sample_weight=weights)
    return {'kind': 'nb', 'model': model, 'supported': supported, 'classes': space['classes']}


def make_mlp(input_dimensions, classes):
    return torch.nn.Sequential(torch.nn.Linear(input_dimensions, 384), torch.nn.GELU(),
        torch.nn.Dropout(.15), torch.nn.Linear(384, len(classes)))


def fit_mlp(space, history, seed, device, epochs=35):
    """Soft-target cross entropy; all positives contribute to the target distribution."""
    rows, targets, totals, supported = aggregated_targets(history, space['classes'], space['input_to_row'])
    torch.manual_seed(seed)
    model = make_mlp(space['dense'].shape[1], space['classes']).to(device)
    features = torch.tensor(space['dense'][rows], device=device)
    labels = torch.tensor(targets / totals[:, None], device=device)
    weights = torch.tensor(np.minimum(np.sqrt(totals), 16), device=device)
    weights /= weights.mean()
    mask = torch.tensor(supported, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.01)
    generator = np.random.default_rng(seed)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(rows))
        loss_total = 0.
        for start in range(0, len(order), 512):
            index = torch.tensor(order[start:start+512], device=device)
            logits = model(features[index]).masked_fill(~mask, -100.)
            loss = (-(labels[index] * torch.nn.functional.log_softmax(logits, dim=1)).sum(dim=1) * weights[index]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(index)
        if epoch in {0, 9, 19, epochs-1}:
            print('Micro MLP epoch', epoch+1, 'inputs', len(rows), 'loss', round(loss_total/len(rows), 4),
                  'seconds', round(time.perf_counter()-started, 1), flush=True)
    model = model.cpu().eval()
    del features, labels, weights, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()
    return {'kind': 'mlp', 'model': model, 'supported': supported, 'classes': space['classes']}


def predict_micro(model, space, frame):
    rows = micro_inputs(frame).key.map(space['input_to_row']).to_numpy(dtype=np.int32)
    if model['kind'] == 'nb':
        probabilities = model['model'].predict_proba(space['sparse'][rows])
        probabilities[:, ~model['supported']] = 0
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    else:
        chunks = []
        mask = torch.tensor(model['supported'])
        with torch.inference_mode():
            for start in range(0, len(rows), 2048):
                logits = model['model'](torch.from_numpy(space['dense'][rows[start:start+2048]])).masked_fill(~mask, -100.)
                chunks.append(torch.softmax(logits, dim=1).numpy())
        probabilities = np.concatenate(chunks)
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1, atol=2e-5)
    return probabilities.astype(np.float32)


def micro_candidate_features(probabilities, candidate_microcats, classes, supported):
    mapping = {int(label): i for i, label in enumerate(classes)}
    columns = np.array([mapping.get(int(label), -1) for label in candidate_microcats])
    valid = columns >= 0
    known = valid.copy()
    known[valid] &= supported[columns[valid]]
    probability = np.full(len(columns), np.nan, np.float32)
    log_probability = probability.copy()
    log_rank = probability.copy()
    probability[known] = probabilities[columns[known]]
    log_probability[known] = np.log(np.maximum(probability[known], 1e-8))
    ranks = rankdata(-probabilities, method='min')
    log_rank[known] = np.log1p(ranks[columns[known]])
    best = np.sort(probabilities)[-2:]
    entropy = float(-(probabilities*np.log(np.maximum(probabilities, 1e-8))).sum()/np.log(max(len(classes), 2)))
    result = np.column_stack([probability, log_probability, log_rank,
        np.full(len(columns), best[-1]), np.full(len(columns), entropy),
        np.full(len(columns), best[-1]-best[-2]), known.astype(np.float32)]).astype(np.float32)
    assert result.shape == (len(columns), len(MICRO_FEATURE_NAMES))
    return result


def classification_metrics(probabilities, frame, gold, item_microcats, classes):
    """Average within each context, then across contexts, as in Recall@50."""
    mapping = {int(label): i for i, label in enumerate(classes)}
    recalls = {k: [] for k in [1, 3, 5, 10]}
    nll, unsupported = [], []
    for p, key in zip(probabilities, frame.context_key):
        labels = [int(item_microcats[i]) for i in gold[key]]
        order = np.argsort(-p, kind='stable')
        for k in recalls:
            selected = set(map(int, classes[order[:k]]))
            recalls[k].append(sum(label in selected for label in labels)/len(labels))
        nll.append(np.mean([-np.log(max(float(p[mapping[label]]), 1e-8)) if label in mapping else -np.log(1e-8) for label in labels]))
        unsupported.append(sum(label not in mapping for label in labels)/len(labels))
    return {**{f'micro_recall{k}': float(np.mean(value)) for k, value in recalls.items()},
            'micro_nll': float(np.mean(nll)), 'unseen_class_fraction': float(np.mean(unsupported))}
