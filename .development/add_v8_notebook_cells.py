"""Expose v8 experiments in the existing research notebook, training opt-in."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'experiments/Avito_neural_experiments.ipynb'
notebook = json.loads(path.read_text(encoding='utf-8'))
if not any('RUN_RANKER_V8' in ''.join(c['source']) for c in notebook['cells']):
    def add(kind,source):
        cell = {'cell_type':kind,'metadata':{},'source':source.splitlines(True),
            'id':hashlib.sha256(source.encode()).hexdigest()[:12]}
        if kind=='code':
            ast.parse(source)
            cell.update(execution_count=None,outputs=[])
        notebook['cells'].append(cell)
    add('markdown','''## Следующие эксперименты после v7

Основной ответ имеет подтверждённый Recall@50 0.897443. Здесь сравниваем новые
конфигурации ранкера, дополнительное обучение E5 с трудными отрицательными
примерами, контрольную дополнительную эпоху без нового loss и полноту нового пула.
Все запуски локальные. По умолчанию читаем отчёты; обучение включается явно.
''')
    add('code','''RUN_RANKER_V8 = False
RUN_HARD_NEGATIVE_V8 = False
RUN_CONTINUATION_V8 = False
RUN_LEARNED_POOL_V8 = False
if RUN_RANKER_V8:
    run_local('ranker_v8.py')
if RUN_HARD_NEGATIVE_V8:
    run_local('query_hard_negative_v8.py')
if RUN_CONTINUATION_V8:
    run_local('query_continuation_v8.py')
if RUN_LEARNED_POOL_V8:
    run_local('learned_pool_v8.py')
show_report('ranker-v8','selection.json')
show_report('ranker-v8','control.json')
show_report('query-hard-negative-v8','training_report.json')
show_report('query-hard-negative-v8','pilot_report.json')
show_report('learned-pool-v8','report.json')
''')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
for cell in notebook['cells']:
    if cell['cell_type']=='code' and 'RUN_RANKER_V8' in ''.join(cell['source']):
        value='''RUN_RANKER_V8 = False
RUN_HARD_NEGATIVE_V8 = False
RUN_CONTINUATION_V8 = False
RUN_LEARNED_POOL_V8 = False
RUN_EXPANDED_POOL_V8 = False
RUN_CE_FEATURE_V8 = False
RUN_COMBINED_POOL_V8 = False

def snapshot_encoder_report():
    # Both encoder pilots share a latest-report convenience path. Preserve each
    # content-addressed report before launching the next pilot.
    directory=ROOT/'artifacts/query-hard-negative-v8'
    report=json.loads((directory/'pilot_report.json').read_text())
    fingerprint=report['fingerprint']
    for name in ['pilot_report.json','training_report.json','development.csv']:
        path=directory/name
        if name=='training_report.json' and json.loads(path.read_text())['fingerprint']!=fingerprint:
            continue  # A cached checkpoint may not regenerate the training report.
        (directory/f'{path.stem}_{fingerprint}{path.suffix}').write_bytes(path.read_bytes())

if RUN_RANKER_V8:run_local('ranker_v8.py')
if RUN_HARD_NEGATIVE_V8:
    run_local('query_hard_negative_v8.py');snapshot_encoder_report()
if RUN_CONTINUATION_V8:
    run_local('query_continuation_v8.py');snapshot_encoder_report()
if RUN_LEARNED_POOL_V8:run_local('learned_pool_v8.py')
if RUN_EXPANDED_POOL_V8:run_local('expanded_pool_rank_v8.py')
if RUN_CE_FEATURE_V8:run_local('ce_feature_v8.py')
if RUN_COMBINED_POOL_V8:run_local('combined_pool_v8.py')
show_report('ranker-v8','selection.json')
show_report('ranker-v8','control.json')
show_report('query-hard-negative-v8','pilot_report_7287a66ad89510fd.json')
show_report('query-hard-negative-v8','pilot_report_5bae8f3260cf556d.json')
show_report('learned-pool-v8','report.json')
show_report('expanded-pool-v8','pilot_report.json')
show_report('expanded-pool-v8','control.json')
show_report('ce-feature-v8','pilot_report.json')
show_report('combined-pool-v8','selection.json')
show_report('combined-pool-v8','control.json')
show_report('combined-pool-v8','reproduction.json')
'''
        ast.parse(value)
        cell['source']=value.splitlines(True)
        cell['outputs']=[];cell['execution_count']=None
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('V8 experiments available in the existing research notebook.')
