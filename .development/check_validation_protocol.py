"""Regression tests for lost gold labels and cross-context leakage."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.inspection_deps'))
import numpy as np
import pandas as pd
from validation_protocol import (freeze_gold, labels_from_gold, history_for_queries,
                                 oof_assignments, history_coverage)

ids = np.array([f'{i:016x}' for i in range(6)])
frame = pd.DataFrame({'context_key': ['a1', 'a2', 'b1', 'c1'],
                      'query_norm': ['alpha', 'alpha', 'beta', 'gamma']})
rows = []
for context, positive_ids in zip(frame.to_dict('records'), [[0, 1], [2], [3], [4]]):
    for item in positive_ids:
        rows.append({**context, 'item_id': ids[item]})
# A duplicate and a train-only item must not change corpus gold labels.
rows.extend([rows[0].copy(), {**frame.iloc[0].to_dict(), 'item_id': 'not-in-corpus'}])
history = pd.DataFrame(rows)
gold = freeze_gold(history, {value: row for row, value in enumerate(ids)})
truths = labels_from_gold(gold, frame)
assert truths == [frozenset([0, 1]), frozenset([2]), frozenset([3]), frozenset([4])]
history.loc[history.item_id == ids[1], 'item_id'] = ids[5]
assert labels_from_gold(gold, frame) == truths, 'History mutation changed gold labels'
try:
    gold['a1'] = frozenset()
except TypeError:
    pass
else:
    raise AssertionError('Gold mapping is mutable')

query = frame.iloc[:1]
fit, cold, known = history_for_queries(history, query, truths[:1], ids,
                                      'held_context', 0, 42)
assert not cold and known.tolist() == [True]
assert 'a1' not in set(fit.context_key) and 'a2' in set(fit.context_key)
fit2, cold2, known2 = history_for_queries(history, query, truths[:1], ids,
                                         'unseen_text', 0, 42)
assert not known2.any() and 'alpha' not in set(fit2.query_norm)
fit3, cold3, _ = history_for_queries(history, query, truths[:1], ids,
                                    'held_context', 1, 42, excluded_contexts=['b1'])
assert cold3 == {ids[0], ids[1]} and 'b1' not in set(fit3.context_key)
assert labels_from_gold(gold, frame) == truths
assert history_coverage(query, truths[:1], fit, ids)['own_context_overlap'] == 0

for mode in ['unseen_text', 'held_context']:
    assignment = oof_assignments(frame, mode, 3, 42)
    assert np.array_equal(assignment, oof_assignments(frame, mode, 3, 42))
    for fold in range(3):
        selected = np.flatnonzero(assignment == fold)
        if not len(selected):
            continue
        q = frame.iloc[selected]
        labels = [truths[i] for i in selected]
        isolated, _, _ = history_for_queries(history, q, labels, ids, mode, .5, 42 + fold)
        assert not set(q.context_key) & set(isolated.context_key)
        if mode == 'unseen_text':
            assert not set(q.query_norm) & set(isolated.query_norm)
assert oof_assignments(frame, 'unseen_text', 3, 42)[0] == oof_assignments(frame, 'unseen_text', 3, 42)[1]
try:
    history_for_queries(history, query, truths[:1], ids, 'held_context', 1.1, 42)
except ValueError:
    pass
else:
    raise AssertionError('Invalid cold fraction accepted')
print('Passed: immutable complete gold, multiple positives, corpus membership, own-context isolation,')
print('known-text preservation, cold-item removal, text-group OOF and deterministic folds.')
