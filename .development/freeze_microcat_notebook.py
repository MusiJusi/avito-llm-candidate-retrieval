"""Freeze the development-selected microcat candidate in the delivered notebook."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/microcat-v6'
manifest=json.loads((CACHE/'manifest.json').read_text())
selection=manifest['selection'];control=manifest['control']
assert hashlib.sha256((ROOT/manifest['answer_file']).read_bytes()).hexdigest()==manifest['answer_sha256']
assert selection['winner']['variant']!='v5'
old=ROOT/'Avito_v0.5.ipynb'
if not old.exists():
    old.write_bytes(subprocess.check_output(['git','show','v0.5.0:Avito.ipynb'],cwd=ROOT))
document=json.loads((ROOT/'Avito_microcat_v6.ipynb').read_text(encoding='utf-8'))
cells=[]
for cell in document['cells']:
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        tree=ast.parse(source)
        last=tree.body[-1] if tree.body else None
        if isinstance(last,ast.Expr) and isinstance(last.value,ast.Call) and isinstance(last.value.func,ast.Name) and last.value.func.id in {'standalone','run_experiment'}:
            continue
        if 'V5_CACHE = CACHE' in source:
            source+='''
# Verify supplied classifier and selector weights, separate from frozen v5.
micro_weights=MICRO_CACHE/'weights_manifest.json'
if micro_weights.exists():
    declared=json.loads(micro_weights.read_text(encoding='utf-8'))
    assert declared['input_sha256']==input_hashes
    for relative,expected in declared['frozen_components'].items():
        assert sha256_file(ROOT/relative)==expected, relative
    if USE_CACHE:
        for relative,expected in declared['cache_components'].items():
            assert sha256_file(ROOT/relative)==expected, relative
'''
        cell=copy.deepcopy(cell)
        cell['source']=source.splitlines(True);cell['outputs']=[];cell['execution_count']=None
    cells.append(cell)

def add(kind,source):
    cell={'cell_type':kind,'source':source.splitlines(True),'metadata':{},
          'id':hashlib.sha256((kind+source).encode()).hexdigest()[:12]}
    if kind=='code':cell.update(outputs=[],execution_count=None)
    cells.append(cell)

cells[0]['source']=['# Кандидатогенерация услуг Авито: v0.6.0\n',
    '\nRestart Kernel → Run All воспроизводит answer_microcat_v6.csv по зафиксированным настройкам.\n',
    'Тексты запросов и фильтров дают дополнительное распределение по microcat.\n',
    'Собственная разметка исключена из OOF-признаков, итоговые модели используют полный train.\n',
    'E5 заморожена; search_category в scoring не используется. Внешних inference API нет.\n',
    'AVITO_RUN_MODEL_SEARCH=1 повторяет исследования, AVITO_REBUILD_CACHE=1 — подготовку и обучение.\n',
    'Веса собственных замороженных компонентов v5 поставляются с кодом и имеют контрольные суммы.\n',
    'Это экспериментальный кандидат: development улучшился, контроль слегка ухудшился.\n',
    'Основной answer.csv от v5 не изменяется. Резерв также сохранён в answer_v0.5.csv.\n']
add('markdown','## Финальное обучение и ответ\n\nНастройки выбраны на development до открытия контрольной выборки.\n'
    'Обычный запуск использует поставляемые веса; при отсутствии финальных весов строит OOF-признаки и обучает модель.\n')
source=(ROOT/'.development/export_microcat_v6.py').read_text(encoding='utf-8')
tree=ast.parse(source)
add('code','\n\n'.join(ast.get_source_segment(source,node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)))+'\n')
add('code','FROZEN_MICRO_SELECTION = '+repr(selection)+'\nFROZEN_MICRO_CONTROL = '+repr(control)+'''
if os.getenv('AVITO_RUN_MODEL_SEARCH','0')=='1':
    standalone()
    run_experiment()
    current_selection=json.loads((MICRO_CACHE/'ranker_selection.json').read_text())
    current_control=json.loads((MICRO_CACHE/'ranker_control.json').read_text())
    export_microcat(current_selection,current_control)
else:
    export_microcat(FROZEN_MICRO_SELECTION,FROZEN_MICRO_CONTROL)
''')
document['cells']=cells
document['metadata']['frozen_micro_selection']=selection['winner']
for cell in cells:
    if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
(ROOT/'Avito_microcat_candidate_v6.ipynb').write_text(json.dumps(document,ensure_ascii=False,indent=1),encoding='utf-8')
print('Frozen microcat notebook:',selection['winner'])
