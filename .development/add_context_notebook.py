"""Expose the independent v9 experiment in the existing research notebook."""
from pathlib import Path
import ast
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'experiments/Avito_neural_experiments.ipynb'
notebook=json.loads(path.read_text(encoding='utf-8'))
marker='RUN_CONTEXT_V9 = False'
if not any(marker in ''.join(c['source']) for c in notebook['cells']):
    description='''## v9: география, семантическая история и расширенные OOF-группы

Самостоятельная реализация общих методов: восстановление географического центра
из разрешённой истории, оценка её надёжности, перенос микрокатегорий и векторов
выбранных объявлений от семантических соседей. Исходники другого кандидата,
модели и ответы не переносятся. `search_category` остаётся вне scoring.

Первый пилот использует 4000 обучающих контекстов, два режима истории и шесть
ранее обученных OOF query-encoder. Сравниваем прежние 51 признак, географию,
соседей и их сочетание на одинаковых широких группах. Замороженный v7-ранкер
служит для mining и не называется OOF-miner. Собственные метки исключаются
из feature-history, валидационные тексты — из обучения encoder.
'''
    source='''RUN_CONTEXT_V9 = False
if RUN_CONTEXT_V9:
    run_local('check_context_evidence.py')
    run_local('context_rank_v9.py')
show_report('context-v9','selection.json')
show_report('context-v9','control.json')
show_report('context-v9','manifest.json')
show_report('context-v9','reproduction.json')
'''
    for kind,text in [('markdown',description),('code',source)]:
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(True),
              'id':hashlib.sha256(text.encode()).hexdigest()[:12]}
        if kind=='code':ast.parse(text);cell.update(execution_count=None,outputs=[])
        notebook['cells'].append(cell)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
for cell in notebook['cells']:
    text=''.join(cell['source'])
    if cell['cell_type']=='markdown' and text.startswith('## v9:'):
        if '12000' not in text:
            text+='\nВторой пилот увеличивает обучение до 12000 контекстов и проверяет меньший\n'
            text+='вес нового ранкера. Дополнительно отдельно проверяется CatBoost YetiRank\n'
            text+='на том же первом наборе OOF-признаков. Новых корневых notebooks не создаём.\n'
            cell['source']=text.splitlines(True)
    elif cell['cell_type']=='code' and marker in text:
        if 'RUN_CONTEXT_SCALE' not in text:
            text=text.replace("show_report('context-v9','selection.json')", """RUN_CONTEXT_SCALE = False
RUN_CONTEXT_CATBOOST = False
if RUN_CONTEXT_SCALE:
    run_local('context_scale_v9.py')
if RUN_CONTEXT_CATBOOST:
    run_local('context_catboost_v9.py')
show_report('context-catboost-v9','selection.json')
show_report('context-v9','selection.json')""")
            ast.parse(text)
            cell['source']=text.splitlines(True)
            cell['execution_count']=None;cell['outputs']=[]
        text=''.join(cell['source'])
        if "run_local('context_scale_v9.py')" in text and "run_local('context_base_scaled_v9.py')" not in text:
            text=text.replace("    run_local('context_scale_v9.py')", "    run_local('context_scale_v9.py')\n    run_local('context_base_scaled_v9.py')")
            ast.parse(text);cell['source']=text.splitlines(True)
            cell['execution_count']=None;cell['outputs']=[]
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('Context v9 available in the existing research notebook; expensive runs opt-in.')
