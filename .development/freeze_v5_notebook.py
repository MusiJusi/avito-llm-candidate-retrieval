"""Freeze development-selected settings; default Run All reproduces the answer.

The full comparison remains available via AVITO_RUN_MODEL_SEARCH=1. Cached final
weights avoid rebuilding sampled training groups. AVITO_REBUILD_CACHE=1 retrains
the final model; the optional mining pilot is supplied as a frozen own component.
"""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/ranking-v5'
manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
selection=json.loads((CACHE/'selection.json').read_text(encoding='utf-8'))
control=json.loads((CACHE/'control_metrics.json').read_text(encoding='utf-8'))
assert selection['winner']['model']!='v4'
assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()==manifest['answer_sha256']
notebook=json.loads((ROOT/'Avito_training_v5.ipynb').read_text(encoding='utf-8'))
notebook=copy.deepcopy(notebook)
notebook['cells'][0]['source']=['# Кандидатогенерация услуг Авито: v0.5.0\n','\n',
    'Самостоятельный notebook: Restart Kernel → Run All. Три Parquet, requirements.txt\n',
    'и приложенные локальные веса. По умолчанию выполняется выбранная на development\n',
    'конфигурация; AVITO_RUN_MODEL_SEARCH=1 повторяет полное сравнение.\n',
    'AVITO_REBUILD_CACHE=1 перестраивает индексы, векторы и финальную модель.\n',
    'Собственные замороженные компоненты перечислены в описании и поставляются с кодом.\n',
    'Никаких внешних API и benchmark-labels. search_category в scoring отсутствует.\n',
    'Предыдущий ответ сохранён в answer_v0.4.csv. Результат — answer.csv.\n']

original='''    data=prepare_v5_training(frame,history_all,'final',miner=miner if winner['model']=='mixed_mined' else None)
    model=fit_v5(data,'final_ranker',mode='unseen_text' if winner['model']=='cold_wide' else None,trees=int(winner['trees']))
    del data
    gc.collect()'''
replacement='''    final_path=CACHE/f'final_ranker_{fingerprint}.joblib'
    if USE_CACHE and final_path.exists():
        model=joblib.load(final_path)
    else:
        data=prepare_v5_training(frame,history_all,'final',miner=miner if winner['model']=='mixed_mined' else None)
        model=fit_v5(data,'final_ranker',mode='unseen_text' if winner['model']=='cold_wide' else None,trees=int(winner['trees']))
        del data
        gc.collect()'''
replacements=0
for cell in notebook['cells']:
    if cell['cell_type']!='code':
        continue
    source=''.join(cell['source'])
    if 'class SemanticIndex:' in source and 'semantic_index = SemanticIndex()' in source:
        # Query/doc embeddings are prepared once in the v5 cache below.
        source=source[:source.index('semantic_index = SemanticIndex()')]+'semantic_index = SemanticIndex()\n'
    if 'INDEX_CACHE = CACHE' in source:
        source+='''
# Verify the declared frozen components before using their scores or mining.
weights_path = CACHE / 'weights_manifest.json'
if weights_path.exists():
    supplied_weights = json.loads(weights_path.read_text(encoding='utf-8'))
    assert supplied_weights['input_sha256'] == input_hashes
    for relative, expected in supplied_weights['frozen_components'].items():
        assert sha256_file(ROOT / relative) == expected, f'Modified frozen component: {relative}'
    if USE_CACHE:
        for name, expected in supplied_weights['final_model'].items():
            assert sha256_file(CACHE / name) == expected, f'Modified final model: {name}'
        for name, expected in supplied_weights['embedding_sha256'].items():
            assert sha256_file(CACHE / name) == expected, f'Modified embedding cache: {name}'
'''
    if original in source:
        source=source.replace(original,replacement)
        replacements+=1
    if source.strip()=='main()':
        source='''# Settings frozen before the held control was opened.
FROZEN_SELECTION = '''+repr(selection)+'''
FROZEN_CONTROL = '''+repr(control)+'''
RUN_MODEL_SEARCH = os.getenv('AVITO_RUN_MODEL_SEARCH','0') == '1'
if RUN_MODEL_SEARCH:
    main()
else:
    frozen_miner = None
    if FROZEN_SELECTION['winner']['model'] == 'mixed_mined':
        pilot_path = CACHE/f'mixed_wide_{fingerprint}.joblib'
        if not pilot_path.exists():
            pilot_data = prepare_v5_training(training,training_history,'evaluation')
            frozen_miner = fit_v5(pilot_data,'mixed_wide')
            del pilot_data
            gc.collect()
        else:
            # Frozen own pilot: used to mine negatives, never as an inference score.
            frozen_miner = joblib.load(pilot_path)
    final_fit_and_export(FROZEN_SELECTION['winner'],frozen_miner,FROZEN_SELECTION,FROZEN_CONTROL)
'''
    cell['source']=source.splitlines(True)
    cell['outputs']=[]
    cell['execution_count']=None
    cell['id']=hashlib.sha256(('code'+source).encode()).hexdigest()[:12]
assert replacements==1
previous=ROOT/'Avito_v0.4.ipynb'
if not previous.exists():
    previous.write_bytes(subprocess.check_output(['git','show','v0.4.0:Avito.ipynb'],cwd=ROOT))
notebook['metadata']['frozen_selection']=selection['winner']
(ROOT/'Avito.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('Frozen v5 notebook; previous v4 notebook preserved.')
