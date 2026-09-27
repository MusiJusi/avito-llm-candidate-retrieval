"""Build a self-contained research notebook; embed code instead of helper imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=json.loads((ROOT/'Avito.ipynb').read_text(encoding='utf-8'))
cells=[]
for cell in base['cells']:
    if cell['cell_type']=='code' and 'FROZEN_SELECTION =' in ''.join(cell['source']):
        continue
    cell=copy.deepcopy(cell)
    if cell['cell_type']=='code':
        cell['outputs']=[];cell['execution_count']=None
    cells.append(cell)

def add(kind,source):
    cell={'cell_type':kind,'metadata':{},'source':source.splitlines(True),
          'id':hashlib.sha256((kind+source).encode()).hexdigest()[:12]}
    if kind=='code':
        cell.update(outputs=[],execution_count=None)
    cells.append(cell)

def definitions(source):
    tree=ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source,node) for node in tree.body
                       if isinstance(node,(ast.FunctionDef,ast.ClassDef)))+'\n'

cells[0]['source']=['# Microcat-сигнал для отбора кандидатов: исследование v6\n',
    '\nСамостоятельный notebook: весь код ниже, без импорта development-модулей.\n',
    'Сначала воспроизводится протокол и инициализация v5; затем сравниваются\n',
    'классификаторы и ранкеры на неизменных кандидатах. answer.csv не изменяется.\n',
    'Вероятности для обучения ранкера строятся вне собственного контекста.\n',
    'search_category, location и item_id не входят в текстовые классификаторы.\n']
add('markdown','## 1. Классификаторы microcat\n\nХешированные символьные/словные признаки и MLP на замороженных E5-векторах.\n'
    'Несколько положительных видов услуг сохраняются как распределение. Неизвестный класс получает признак отсутствия данных, а не жёсткий фильтр.\n')
core=(ROOT/'.development/microcat_models.py').read_text(encoding='utf-8')
add('code',core)
micro=(ROOT/'.development/microcat_v6.py').read_text(encoding='utf-8')
initial=micro[micro.index('V5_CACHE = CACHE'):micro.index('def prepare_micro_space')]
initial=initial.replace("hashlib.sha256(Path(driver_file).read_bytes()).hexdigest()",repr(hashlib.sha256(micro.encode()).hexdigest()))
initial=initial.replace("hashlib.sha256((ROOT/'.development/microcat_models.py').read_bytes()).hexdigest()",repr(hashlib.sha256(core.encode()).hexdigest()))
add('code','from threadpoolctl import threadpool_limits\nthreadpool_limits(limits=8)\n'+initial)
add('code',definitions(micro))
add('code','standalone()\n')
add('markdown','## 2. Изолированные OOF-признаки и ablation ранкера\n\n'
    'Одинаковые кандидаты, положительные метки, веса и параметры LambdaRank.\n'
    'Сравниваются добавления NB, MLP и обоих сигналов. Контроль открывается после выбора на development;\n'
    'его ограничение из-за прошлых экспериментов и frozen priors v4 отражено в отчёте.\n')
rank=(ROOT/'.development/microcat_rank_v6.py').read_text(encoding='utf-8')
initial=rank[rank.index('RANK_FP ='):rank.index('def training_micro_features')]
initial=initial.replace('hashlib.sha256(RANK_DRIVER.read_bytes()).hexdigest()',repr(hashlib.sha256(rank.encode()).hexdigest()))
add('code',initial)
add('code',definitions(rank))
add('code',"# Fresh research runs can rebuild the old sampled data/pools from raw inputs.\n"
    "if not (V5_CACHE/f'evaluation_mined_{V5_FP}.joblib').exists():\n"
    "    pilot_path=V5_CACHE/f'mixed_wide_{V5_FP}.joblib'\n"
    "    if pilot_path.exists():\n        pilot=joblib.load(pilot_path)\n"
    "    else:\n        initial_data=prepare_v5_training(training,training_history,'evaluation')\n"
    "        pilot=fit_v5(initial_data,'mixed_wide')\n        del initial_data\n"
    "    mined_data=prepare_v5_training(training,training_history,'evaluation',miner=pilot)\n"
    "    del mined_data,pilot\n    gc.collect()\n"
    "run_experiment()\n")
base['cells']=cells
base['metadata'].pop('frozen_selection',None)
destination=ROOT/'Avito_microcat_v6.ipynb'
for cell in cells:
    if cell['cell_type']=='code':
        ast.parse(''.join(cell['source']))
destination.write_text(json.dumps(base,ensure_ascii=False,indent=1),encoding='utf-8')
print('Written',destination,'code cells',sum(c['cell_type']=='code' for c in cells))
