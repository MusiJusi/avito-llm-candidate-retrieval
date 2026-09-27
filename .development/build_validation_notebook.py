"""Build a self-contained, bounded validation notebook without model fitting.

The historical context recipe is embedded from v0.4.0 to preserve query identities.
New label/history contracts are embedded from validation_protocol.py. The produced
notebook imports no development helper and never writes answer.csv.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
old = json.loads(subprocess.check_output(['git', 'show', 'v0.4.0:Avito.ipynb'], cwd=ROOT).decode('utf-8'))
old_code = [''.join(cell['source']) for cell in old['cells'] if cell['cell_type'] == 'code']
nodes = {}
for source in old_code:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = ast.get_source_segment(source, node)
protocol = (ROOT / '.development/validation_protocol.py').read_text(encoding='utf-8')
cells = []


def cell(kind, source):
    source = source.strip() + '\n'
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(True),
             'id': hashlib.sha256((kind + source).encode()).hexdigest()[:12]}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell('markdown', '''# Валидация v5: полные метки и два режима истории

Этот ноутбук подготавливает протокол следующего обучения. Он не обучает модели,
не выбирает параметры по скрытому тесту и не меняет отправленный answer.csv.
Входы — три предоставленных Parquet. Все функции находятся в этом ноутбуке.

Полные положительные метки фиксируются до очистки истории. Для нового текста
удаляем все его контексты; для известного текста удаляем только удержанные
контексты. Фактическую доступность истории измеряем, а не предполагаем.
`search_category` участвует только в идентичности исходного контекста, как в v4;
признаки scoring в этом ноутбуке не строятся.''')
cell('code', '''from pathlib import Path
import gc, hashlib, html, json, re, sys, unicodedata
from types import MappingProxyType
ROOT = Path.cwd()
if not (ROOT / 'train.parquet').exists():
    raise FileNotFoundError('Запустите notebook из папки с тремя Parquet')
if (ROOT / '.inspection_deps').is_dir():
    sys.path.insert(0, str(ROOT / '.inspection_deps'))
import numpy as np
import pandas as pd
SEED = 260926
CONFIG = {'validation_texts': 1400, 'cold_item_fraction': .9, 'development_fraction': .5}
RANKER_CONFIG = {'audit_queries': 600, 'fresh_audit_queries': 600, 'training_queries': 4000}
QUERY_COLS = ['search_query', 'search_location_id', 'search_is_delivery_search',
              'search_infm_params_text', 'search_category']
CACHE = ROOT / 'artifacts' / 'validation-v5'
CACHE.mkdir(parents=True, exist_ok=True)
train = pd.read_parquet(ROOT / 'train.parquet', columns=QUERY_COLS +
                       ['item_id', 'item_location_id', 'item_microcat_id'])
queries = pd.read_parquet(ROOT / 'benchmark_queries.parquet')
items = pd.read_parquet(ROOT / 'benchmark_items.parquet', columns=['item_id'])
items = items.sort_values('item_id').reset_index(drop=True)
ITEM_IDS = items.item_id.to_numpy(dtype=str)
ITEM_TO_ROW = {value: row for row, value in enumerate(ITEM_IDS)}
assert items.item_id.is_unique and queries.query_id.is_unique
for frame in [train, queries]:
    for column in ['search_query', 'search_infm_params_text']:
        frame[column] = frame[column].fillna('').astype('string[pyarrow]')
''')
cell('code', nodes['normalize_text'] + '''

for frame in [train, queries]:
    frame['query_norm'] = frame.search_query.map(normalize_text).astype('string[pyarrow]')
    frame['context_key'] = [hashlib.sha256(json.dumps([str(v) for v in row],
        ensure_ascii=False).encode()).hexdigest()[:24]
        for row in frame[QUERY_COLS].itertuples(index=False, name=None)]
''' + '\n\n'.join(nodes[name] for name in ['select_query_contexts', 'query_labels', 'purge_history']))
cell('markdown', '''## Сохранение прежних контекстов для сравнения

Следующие две ячейки воспроизводят историческую выборку v4. Вызовы старого
query_labels нужны только для воспроизведения прежнего разбиения и диагностики
потерянных меток. При дальнейшей подготовке используем исключительно полный gold.
Все ранее просмотренные 3400 контекстов считаем development; независимым новым
тестом эта выборка больше не является.''')
cell('code', old_code[3])
historical = old_code[5]
cell('code', historical[historical.index('audit_queries ='):historical.index('started = time.perf_counter()')] + '''

old_direct_texts = set(training_queries.query_norm) | set(original_training_queries.query_norm)
old_direct_ids = {ITEM_IDS[i] for truth in query_labels(ranker_history, training_queries) for i in truth}
overlap_texts = set(ranker_history.loc[ranker_history.item_id.isin(old_direct_ids), 'query_norm'])
audit_source = ranker_history[~ranker_history.query_norm.isin(old_direct_texts | overlap_texts)]
new_audit = select_query_contexts(audit_source, 800, SEED + 702)
old_new_audit_labels = query_labels(ranker_history, new_audit)
development = pd.concat([validation[[*QUERY_COLS, 'query_norm', 'context_key']],
                         audit_queries, fresh_queries, new_audit], ignore_index=True)
assert development.context_key.is_unique
''')
cell('markdown', '''## Независимые gold-метки и очищаемая история

Gold содержит все наблюдённые положительные объявления доступного корпуса.
Неизменяемая структура защищает от случайного удаления меток при purge истории.
Без времени событий мы не можем независимо проверить повторный выбор в полностью
совпадающем контексте; показываем его долю отдельно, не выдаём её за честный holdout.''')
cell('code', protocol + '''

gold = freeze_gold(history_all, ITEM_TO_ROW)
development_truths = labels_from_gold(gold, development)
assert all(development_truths)
restored_new = labels_from_gold(gold, new_audit)
label_changes = {'old_new_audit_positives': sum(map(len, old_new_audit_labels)),
    'full_new_audit_positives': sum(map(len, restored_new)),
    'affected_new_audit_contexts': sum(a != b for a, b in zip(old_new_audit_labels, restored_new))}
assert label_changes == {'old_new_audit_positives': 815, 'full_new_audit_positives': 822,
                         'affected_new_audit_contexts': 6}
known_text = queries.query_norm.isin(history_all.query_norm)
known_context = queries.context_key.isin(history_all.context_key)
target = {'unseen_text_fraction': float((~known_text).mean()),
    'known_text_new_context_fraction': float((known_text & ~known_context).mean()),
    'exact_context_fraction': float(known_context.mean()),
    'exact_context_evaluation_limitation': 'No timestamps or independent repeated events'}
assert np.isclose(sum(target[k] for k in ['unseen_text_fraction',
    'known_text_new_context_fraction', 'exact_context_fraction']), 1)
print('Исправление меток:', json.dumps(label_changes, ensure_ascii=False))
print('Режимы бенчмарка:', json.dumps(target, ensure_ascii=False))
profiles = []
for mode in ['unseen_text', 'held_context']:
    for fraction in [0., .5, .9, 1.]:
        fit, cold, _ = history_for_queries(history_all, development,
            development_truths, ITEM_IDS, mode, fraction, SEED + 801)
        profile = history_coverage(development, development_truths, fit, ITEM_IDS)
        profile.update(mode=mode, cold_fraction=fraction, cold_items=len(cold))
        profiles.append(profile)
        assert profile['own_context_overlap'] == 0
        del fit
print(pd.DataFrame(profiles).to_string(index=False))
''')
cell('markdown', '''## Подготовка следующего обучающего набора

Контексты для прямого обучения выбираем по полным gold-меткам, а не по очищенной
истории. Иначе удаление item из вспомогательной истории теряет ещё и обучающий пример.
Тексты всех development-запросов исключаем из нового прямого обучения.
История обучения также исключает эти тексты и выбранные cold items.

Для каждого контекста предусмотрены два режима OOF. В одном группируем по тексту,
в другом по полному контексту. Второй режим может фактически стать неизвестным
после очистки: это отмечено отдельным флагом known_text, необходимым при обучении.
Ранги и hard negatives будут рассчитываться следующим этапом по полному пулу.
''')
cell('code', nodes['context_sample'] + '''

# Reserve text groups not used for direct supervised fitting in v4. Reconstruct
# that list locally rather than depend on an untracked cached parquet file.
EXPERIMENT_CONFIG = {'contexts_per_text': 4}
old_clean, _ = purge_history(ranker_history, new_audit, old_new_audit_labels, SEED + 703)
old_expanded = context_sample(old_clean, 20000, SEED + 704, training_queries)
previous_direct_texts = set(old_expanded.query_norm) | set(development.query_norm)
control_source = history_all[~history_all.query_norm.isin(previous_direct_texts)]
held_control = select_query_contexts(control_source, 600, SEED + 805)
held_control_truths = labels_from_gold(gold, held_control)
assert len(held_control) == 600 and all(held_control_truths)
assert not set(held_control.query_norm) & previous_direct_texts
evaluation_contexts = pd.concat([development, held_control], ignore_index=True)
evaluation_truths = development_truths + held_control_truths
training_history, development_cold, _ = history_for_queries(history_all,
    evaluation_contexts, evaluation_truths, ITEM_IDS, 'unseen_text', .9, SEED + 802)
candidate_contexts = history_all[history_all.context_key.isin(gold)].drop_duplicates('context_key')
candidate_contexts = candidate_contexts[~candidate_contexts.query_norm.isin(evaluation_contexts.query_norm)]
candidate_contexts = candidate_contexts[[*QUERY_COLS, 'query_norm', 'context_key']].sort_values('context_key').reset_index(drop=True)
candidate_truths = labels_from_gold(gold, candidate_contexts)
assert all(candidate_truths)
assert not set(candidate_contexts.query_norm) & set(evaluation_contexts.query_norm)
fold_profiles, assignment_rows = [], []
for mode in ['unseen_text', 'held_context']:
    assignments = oof_assignments(candidate_contexts, mode, 3, SEED + 803)
    for fold in range(3):
        positions = np.flatnonzero(assignments == fold)
        q = candidate_contexts.iloc[positions]
        truth = [candidate_truths[i] for i in positions]
        fit, cold, actual_known = history_for_queries(training_history, q, truth,
            ITEM_IDS, mode, .9, SEED + 804 + fold)
        profile = history_coverage(q, truth, fit, ITEM_IDS)
        profile.update(mode=mode, fold=fold, cold_items=len(cold), history_pairs=len(fit))
        fold_profiles.append(profile)
        assignment_rows.extend({'context_key': key, 'mode': mode, 'fold': fold,
            'actual_known_text': bool(known)} for key, known in zip(q.context_key, actual_known))
        assert profile['own_context_overlap'] == 0
        if mode == 'unseen_text':
            assert profile['known_text_fraction'] == 0
        del fit
        gc.collect()
prepared = candidate_contexts.copy()
prepared['relevant_item_ids'] = [' '.join(ITEM_IDS[i] for i in sorted(truth)) for truth in candidate_truths]
prepared.to_parquet(CACHE / 'training_contexts.parquet', index=False)
prepared_dev = development.copy()
prepared_dev['relevant_item_ids'] = [' '.join(ITEM_IDS[i] for i in sorted(truth)) for truth in development_truths]
prepared_dev.to_parquet(CACHE / 'development_contexts.parquet', index=False)
prepared_control = held_control.copy()
prepared_control['relevant_item_ids'] = [' '.join(ITEM_IDS[i] for i in sorted(truth)) for truth in held_control_truths]
prepared_control.to_parquet(CACHE / 'held_control_contexts.parquet', index=False)
pd.DataFrame(assignment_rows).to_parquet(CACHE / 'oof_assignments.parquet', index=False)
pd.DataFrame(profiles).to_csv(CACHE / 'history_regimes.csv', index=False)
pd.DataFrame(fold_profiles).to_csv(CACHE / 'oof_history_audit.csv', index=False)
report = {'stage': 'validation-v5', 'gold_source': 'complete unpurged train, corpus positives only',
    'gold_contexts': len(gold), 'gold_positive_pairs': sum(map(len, gold.values())),
    'label_correction': label_changes, 'benchmark_regimes': target,
    'development_contexts': len(development), 'development_role': 'previously viewed development',
    'held_control_contexts': len(held_control), 'held_control_direct_text_overlap': 0,
    'held_control_limitation': 'Frozen v4 priors may have used these interactions in auxiliary history; strict validation requires refitting affected priors',
    'training_contexts': len(candidate_contexts),
    'training_texts': int(candidate_contexts.query_norm.nunique()),
    'training_positives': sum(map(len, candidate_truths)),
    'oof_audits': fold_profiles, 'evaluation_history_profiles': profiles,
    'new_models_fitted': False, 'answer_csv_modified': False,
    'next_stage': 'full-pool retrieval, stronger hard negatives, mixed-regime ranker training'}
(CACHE / 'protocol.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k not in ['oof_audits', 'evaluation_history_profiles']},
                 ensure_ascii=False, indent=2))
print(pd.DataFrame(fold_profiles).to_string(index=False))
''')
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)',
                                      'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.14.6'}}}
(ROOT / 'Avito_validation_v5.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Created standalone validation notebook:', len(cells), 'cells')
