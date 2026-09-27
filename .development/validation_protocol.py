"""Validation contracts for v5; the notebook embeds these functions verbatim.

Gold labels come from the complete interaction table. History is a separate,
mutable view: removing information from it must never remove evaluation labels.
"""
from types import MappingProxyType

import numpy as np
import pandas as pd


def freeze_gold(full_pairs, item_to_row):
    """Snapshot all observed positives in the searchable corpus, once.

    Both the mapping and its values are immutable. Corpus membership is the only
    label filter; geography and historical purges do not affect the gold set.
    """
    eligible = full_pairs.loc[full_pairs.item_id.isin(item_to_row),
                              ['context_key', 'item_id']].drop_duplicates()
    grouped = eligible.groupby('context_key', sort=True).item_id.agg(
        lambda values: frozenset(item_to_row[value] for value in values))
    return MappingProxyType(grouped.to_dict())


def labels_from_gold(gold, query_frame):
    """Return immutable labels in query order; never accept a history argument."""
    return [gold.get(key, frozenset()) for key in query_frame.context_key]


def history_for_queries(history, query_frame, truths, item_ids, mode,
                        cold_fraction, seed, excluded_contexts=(),
                        excluded_texts=(), excluded_items=()):
    """Build an isolated feature history for a specified evaluation regime.

    unseen_text removes every context of the held text. held_context removes
    own context interactions but permits other contexts of the same text.
    Without timestamps, this does not evaluate repeats of the identical context.
    """
    if mode not in {'unseen_text', 'held_context'}:
        raise ValueError(f'Unknown history mode: {mode}')
    if not 0 <= cold_fraction <= 1:
        raise ValueError('cold_fraction must be in [0, 1]')
    if len(query_frame) != len(truths):
        raise ValueError('One gold set is required per query')
    positive_ids = np.array(sorted({item_ids[i] for truth in truths for i in truth}),
                            dtype=str)
    random = np.random.default_rng(seed)
    cold = set(random.choice(positive_ids, int(cold_fraction * len(positive_ids)),
                             replace=False))
    blocked_contexts = set(query_frame.context_key) | set(excluded_contexts)
    blocked_texts = set(excluded_texts)
    if mode == 'unseen_text':
        blocked_texts.update(query_frame.query_norm)
    blocked_items = cold | set(excluded_items)
    fit = history.loc[~history.context_key.isin(blocked_contexts)
                      & ~history.query_norm.isin(blocked_texts)
                      & ~history.item_id.isin(blocked_items)].copy()
    assert not blocked_contexts & set(fit.context_key)
    assert not blocked_items & set(fit.item_id)
    assert not blocked_texts & set(fit.query_norm)
    known = query_frame.query_norm.isin(fit.query_norm).to_numpy(dtype=bool)
    if mode == 'unseen_text':
        assert not known.any()
    return fit, cold, known


def oof_assignments(query_frame, mode, folds=3, seed=260926):
    """Group by text for cold queries, by context for held-context queries.

    A held context includes all its positives and duplicate rows. Different
    contexts of one text may appear in the fit history in the second regime.
    """
    if mode not in {'unseen_text', 'held_context'}:
        raise ValueError(f'Unknown history mode: {mode}')
    if folds < 2:
        raise ValueError('At least two folds are required')
    column = 'query_norm' if mode == 'unseen_text' else 'context_key'
    values = np.sort(query_frame[column].unique().astype(str))
    random = np.random.default_rng(seed)
    shuffled = values[random.permutation(len(values))]
    mapping = {key: position % folds for position, key in enumerate(shuffled)}
    return query_frame[column].map(mapping).to_numpy(dtype=np.int32)


def history_coverage(query_frame, truths, fit, item_ids):
    """Measure actual availability rather than infer it from a regime's name."""
    known = query_frame.query_norm.isin(fit.query_norm).to_numpy(dtype=bool)
    observed_items = set(fit.item_id)
    fractions = [sum(item_ids[i] in observed_items for i in truth) / len(truth)
                 for truth in truths if truth]
    return {'contexts': len(query_frame), 'positive_pairs': sum(map(len, truths)),
            'known_text_fraction': float(known.mean()) if len(known) else None,
            'positive_item_seen_macro_fraction': float(np.mean(fractions)) if fractions else None,
            'own_context_overlap': len(set(query_frame.context_key) & set(fit.context_key))}
